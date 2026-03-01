import pandas as pd

df = pd.read_csv('data/synthetic_orders.csv')

print('=' * 60)
print('ANALYSIS SUMMARY')
print('=' * 60)
print(f'Avg rider wait: {df["actual_rider_wait_minutes"].mean():.3f} minutes')
print(f'Avg naive error: {df["naive_kpt_error"].mean():.3f} minutes')
print(f'% rider wait >5 min: {(df["actual_rider_wait_minutes"] > 5).mean()*100:.1f}%')
print()
print('Key statistics:')
print(f'Total orders: {len(df):,}')
print(f'Restaurants: {df["restaurant_id"].nunique()}')
print(f'Avg true KPT: {df["true_kpt_minutes"].mean():.2f} minutes')
print(f'Avg POS signal error: {abs(pd.to_datetime(df["pos_ticket_cleared_time"]) - pd.to_datetime(df["actual_ready_time"])).dt.total_seconds().div(60).abs().mean():.2f} minutes')
print()
print('Signal comparison:')
biased = df[df['honest_merchant'] == False]
honest = df[df['honest_merchant'] == True]
print(f'Biased merchants (n={len(biased)}): FOR delay avg = {((pd.to_datetime(biased["for_button_time"]) - pd.to_datetime(biased["actual_ready_time"])).dt.total_seconds().div(60).mean()):.2f} min')
print(f'Honest merchants (n={len(honest)}): FOR delay avg = {((pd.to_datetime(honest["for_button_time"]) - pd.to_datetime(honest["actual_ready_time"])).dt.total_seconds().div(60).mean()):.2f} min')
print('=' * 60)
