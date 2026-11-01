"""Construct Retailrocket queries using strictly earlier behavioral features."""
from pathlib import Path
from collections import defaultdict
import zipfile, json, hashlib
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
MS_HOUR = 3600000

def build_queries(events):
    cart_users = set(events.loc[events.event == 'addtocart', 'visitorid'].astype(int))
    users = {}
    pairs = {}
    items = {}
    rows = []

    def update(t, v, kind, item):
        typ = {'view': 0, 'addtocart': 1, 'transaction': 2}[kind]
        stat = items.get(item)
        if stat is None:
            stat = [0, 0, 0, t, t]
            items[item] = stat
        stat[typ] += 1
        stat[4] = t
        if v not in cart_users:
            return
        s = users.get(v)
        if s is None:
            s = [0, 0, 0, t, t, -1, t, 0, 0, 0]
            users[v] = s
        if t - s[4] > 1800000:
            s[6] = t
            s[7:10] = [0, 0, 0]
        s[typ] += 1
        s[7 + typ] += 1
        s[4] = t
        if typ == 2:
            s[5] = t
        key = (v, item)
        p = pairs.get(key)
        if p is None:
            p = [0, 0, 0, t, t, -1, -1, -1]
            pairs[key] = p
        p[typ] += 1
        p[4] = t
        p[5 + typ] = t
    pending = []
    last_t = None
    for row in events[['timestamp', 'visitorid', 'event', 'itemid', 'raw_event_row']].itertuples(index=False, name=None):
        t, v, kind, item, raw_id = row
        t = int(t)
        v = int(v)
        item = int(item)
        if t != last_t:
            for a in pending:
                update(*a)
            pending = []
            last_t = t
        if kind == 'addtocart':
            u = users.get(v, [0, 0, 0, t, t, -1, t, 0, 0, 0])
            p = pairs.get((v, item), [0, 0, 0, t, t, -1, -1, -1])
            it = items.get(item, [0, 0, 0, t, t])
            new_session = t - u[4] > 1800000
            dt = pd.Timestamp(t, unit='ms', tz='UTC')
            s = {'source_id': f'events.csv:{raw_id}', 'raw_event_row': raw_id, 'timestamp': t, 'visitorid': v, 'itemid': item, 'hour': dt.hour, 'weekday': dt.dayofweek, 'query_day': dt.strftime('%Y-%m-%d'), 'visitor_views': u[0], 'visitor_carts': u[1], 'visitor_purchases': u[2], 'visitor_age_hours': (t - u[3]) / MS_HOUR, 'visitor_idle_minutes': (t - u[4]) / 60000, 'prior_buyer': int(u[2] > 0), 'session_views': 0 if new_session else u[7], 'session_carts': 0 if new_session else u[8], 'session_purchases': 0 if new_session else u[9], 'session_age_minutes': 0 if new_session else (t - u[6]) / 60000, 'pair_views': p[0], 'pair_carts': p[1], 'pair_purchases': p[2], 'pair_age_minutes': (t - p[3]) / 60000, 'pair_idle_minutes': (t - p[4]) / 60000, 'item_views': it[0], 'item_carts': it[1], 'item_purchases': it[2], 'item_age_hours': (t - it[3]) / MS_HOUR, 'item_cart_view_ratio': it[1] / max(it[0], 1), 'item_purchase_cart_ratio': it[2] / max(it[1], 1), 'visitor_purchase_cart_ratio': u[2] / max(u[1], 1)}
            rows.append(s)
        pending.append((t, v, kind, item))
    return pd.DataFrame(rows)

def main():
    raw = H / 'raw'
    out = H / 'processed'
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(raw / 'RetailRocket.zip') as z:
        payload = z.read('events.csv')
        (raw / 'retailrocket_events.csv').write_bytes(payload)
        events = pd.read_csv(raw / 'retailrocket_events.csv')
    events['raw_event_row'] = np.arange(len(events))
    events = events.sort_values(['timestamp', 'raw_event_row'], kind='stable').reset_index(drop=True)
    print('RAW_EVENTS', len(events), 'TYPES', events.event.value_counts().to_dict(), flush=True)
    q = build_queries(events)
    q = q.sort_values(['timestamp', 'raw_event_row'], kind='stable').drop_duplicates(['visitorid', 'itemid', 'query_day'], keep='first').reset_index(drop=True)
    tx = events[events.event == 'transaction'].groupby(['visitorid', 'itemid']).timestamp.apply(lambda s: s.to_numpy())
    lookup = tx.to_dict()
    next_time = []
    same_time = []
    for v, i, t in q[['visitorid', 'itemid', 'timestamp']].itertuples(index=False, name=None):
        arr = lookup.get((v, i))
        if arr is None:
            next_time.append(-1)
            same_time.append(False)
            continue
        j = np.searchsorted(arr, t, side='left')
        same_time.append(j < len(arr) and arr[j] == t)
        k = np.searchsorted(arr, t, side='right')
        next_time.append(int(arr[k]) if k < len(arr) else -1)
    q['next_same_item_purchase_ms'] = next_time
    q['same_timestamp_transaction'] = same_time
    ambiguous = int(q.same_timestamp_transaction.sum())
    q = q[~q.same_timestamp_transaction].copy()
    q.to_pickle(out / 'retail_causal_queries.pkl')
    audit = {'raw_events': len(events), 'event_counts': events.event.value_counts().to_dict(), 'raw_event_sha256': hashlib.sha256(payload).hexdigest(), 'raw_archive_sha256': hashlib.sha256((raw / 'RetailRocket.zip').read_bytes()).hexdigest(), 'one_query_per_visitor_item_UTCday': True, 'same_timestamp_purchase_queries_excluded': ambiguous, 'queries_after_dedup_and_tie_exclusion': len(q), 'feature_boundary': 'All behavioral feature counts use raw event timestamps strictly earlier than the query. Same-timestamp events are withheld as a group.', 'business_target': 'Same visitor buys the same item after adding it to cart, within the fixed horizon. Not causal response to discounts; no revenue uplift claimed.', 'end_of_log_ms': int(events.timestamp.max()), 'target_only_fields': ['next_same_item_purchase_ms', 'transactionid', 'feedback_clock', 'label']}
    (out / 'retail_construction_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    cat = json.loads((out / 'catalog.json').read_text())
    cat = [x for x in cat if x['family'] != 'RetailRocket']
    compact = ['hour', 'weekday', 'prior_buyer', 'session_views', 'session_carts', 'session_purchases', 'pair_views', 'pair_carts', 'visitor_idle_minutes', 'session_age_minutes', 'pair_idle_minutes', 'item_cart_view_ratio', 'item_purchase_cart_ratio']
    full = compact + ['visitor_views', 'visitor_carts', 'visitor_purchases', 'visitor_age_hours', 'pair_purchases', 'pair_age_minutes', 'item_views', 'item_carts', 'item_purchases', 'item_age_hours', 'visitor_purchase_cart_ratio']
    unchanged = {'hour', 'weekday', 'prior_buyer', 'item_cart_view_ratio', 'item_purchase_cart_ratio', 'visitor_purchase_cart_ratio'}
    for hours in [1, 24]:
        eligible = q[q.timestamp + hours * MS_HOUR <= events.timestamp.max()].copy()
        eligible['label'] = ((eligible.next_same_item_purchase_ms > eligible.timestamp) & (eligible.next_same_item_purchase_ms <= eligible.timestamp + hours * MS_HOUR)).astype(int)
        eligible['clock'] = eligible.timestamp.astype('int64') * 1000000
        eligible['feedback_clock'] = np.where(eligible.label == 1, eligible.next_same_item_purchase_ms, eligible.timestamp + hours * MS_HOUR).astype('int64') * 1000000
        for scope in ['all', 'first_item_cart', 'experienced_shopper']:
            d = eligible.copy()
            if scope == 'first_item_cart':
                d = d[d.pair_carts == 0].copy()
            if scope == 'experienced_shopper':
                d = d[d.visitor_carts >= 1].copy()
            if len(d) < 1500:
                continue
            for version, features in [('compact', compact), ('full', full)]:
                data = d.copy()
                nums = []
                for f in features:
                    if f in unchanged:
                        nums.append(f)
                    else:
                        col = 'log1p_' + f
                        data[col] = np.log1p(data[f])
                        nums.append(col)
                name = f'RetailRocket_cart{hours}h_{scope}_{version}'
                data = data.sort_values(['clock', 'source_id'], kind='stable').reset_index(drop=True)
                data['stream_row'] = np.arange(len(data))
                cut = int(len(data) * 0.6)
                while cut > 0 and data.clock.iloc[cut] == data.clock.iloc[cut - 1]:
                    cut -= 1
                data.to_pickle(out / f'{name}.pkl')
                info = {'name': name, 'family': 'RetailRocket', 'n': len(data), 'outer_cut': cut, 'positive_rate': float(data.label.mean()), 'numeric_features': nums, 'categorical_features': [], 'sizes': [2400, 6000], 'availability': 'OBSERVED_PURCHASE_OR_FIXED_HORIZON', 'horizon_hours': hours, 'scope': scope, 'feature_version': version, 'target': f'Same visitor/item purchase strictly after cart and within{hours}hours', 'raw_file': 'retailrocket_events.csv', 'business_use': 'Monitor cart-conversion predictions for prioritizing follow-up and identifying conversion changes', 'source_order': 'Real event timestamp; first visitor-item cart per UTCday; no class balancing or shuffled replay', 'feature_boundary': audit['feature_boundary'], 'processed_sha256': hashlib.sha256((out / f'{name}.pkl').read_bytes()).hexdigest()}
                cat.append(info)
                print(name, len(data), 'positive', round(data.label.mean(), 4), flush=True)
    (out / 'catalog.json').write_text(json.dumps(cat, ensure_ascii=False, indent=2))
    print('TOTAL_VARIANTS', len(cat), flush=True)
if __name__ == '__main__':
    main()
