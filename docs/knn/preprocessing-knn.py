import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("docs/knn/booking.csv")

df = df.drop(columns=["Booking_ID", "date of reservation"])

categorical_cols = ['type of meal', 'room type', 'market segment type']

encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_features = encoder.fit_transform(df[categorical_cols])

encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

df_final = df.drop(categorical_cols, axis=1)
df_final = pd.concat([df_final, encoded_df], axis=1)

