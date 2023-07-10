from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split





data = pd.read_csv("data/games1.csv")
categorical_columns = ["opening"]

encoder = OneHotEncoder(sparse=False)
encoder.fit(data[categorical_columns])

encoded_columns = encoder.transform(data[categorical_columns])

encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(categorical_columns))

final_df = pd.concat([data, encoded_df], axis=1)

final_df = final_df.drop(columns=["Unnamed: 0.1", "Unnamed: 0", "pgn", "opening"])

X = final_df.drop("average_rating", axis=1)
Y = final_df["average_rating"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = X_train.shape[1:]) )
model.add(tf.keras.layers.Dense(256, activation = "relu"))
model.add(tf.keras.layers.Dense(2301))

model.compile(optimizer = "adam", loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ["accuracy"])
model.fit(X_train, Y_train, epochs = 500)

print(model.evaluate(X_test, Y_test))

model.save("GTEMODELV1.h5")