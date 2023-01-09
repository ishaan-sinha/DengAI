import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

compare_df = pd.read_csv('compare.csv')

print(mean_squared_error(compare_df['actual'], compare_df['predicted_mean'], squared=False))
print(mean_absolute_error(compare_df['actual'], compare_df['predicted_mean']))
print(r2_score(compare_df['actual'], compare_df['predicted_mean']))
