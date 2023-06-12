# UnbiasedGBM
repository for Unbiased Gradient Boosting Decision Tree with Unbiased Feature Importance


## model

**UPD: 6.12**

This part is our classification/regression model.

```bash
bash compile.sh
python3 example.py
```

UnbiasedBoost

- `losstool`: 'logloss' for classification or 'MSE' for regression
- `n_est`: number of estimators
- `min_leaf`: minimum number of instances at leaf node
- `lr`: learning rate
- `n_leaf`: number of leaves per tree.

.fit

- `df`: dataframe. categorical features should input as int, and numerical feature should input as float.
- `label`
-  `testset`: tuple(metric, df_test, df_label) (see example.py)
- `return_pred`: whether to return prediction of testset

.predict

- `df`: dataframe.

.calc_self_imp

This method will return the importance in training stage, different from our post-hoc method.

## UnbiasedGain

This part is a post-hoc method.

**Support XGBoost LightGBM**

```python
UnbiasedGain.calc_gain(model, dataT, labelT, dataV, labelV, losstool)
```

### Example Code

```python
seed = 998244353
model = (lgb.LGBMRegressor if task=='regression' else lgb.LGBMClassifier)(random_state=seed, learning_rate=1, n_estimators=5)
model.fit(X_train, y_train.values)
pred = model.predict(X_test) if task=='regression' else model.predict_proba(X_test)[:,1]
print(model.feature_importances_)
losstool = UnbiasedGain.MSE_tool() if task=='regression' else UnbiasedGain.logloss_tool()
UnbiasedGain.calc_gain(model, X_train, y_train, X_test, y_test, losstool)
```

