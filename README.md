# UnbiasedGBM
repository for Unbiased Gradient Boosting Decision Tree with Unbiased Feature Importance


## UnbiasedGain

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