#
# Busque los mejores parametros de un modelo knn para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Considere diferentes valores para la cantidad de vecinos
#

# importacion de librerias
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

from homework.src.__internals.calculate_metrics import calculate_metrics
from homework.src.__internals.prepare_data import prepare_data
from homework.src.__internals.print_metrics import print_metrics
from homework.src.__internals.save_model_if_better import save_model_if_better


x_train, x_test, y_train, y_test = prepare_data(
    file_path="data/winequality-red.csv",
    test_size=0.25,
    random_state=123456,
)

estimator = KNeighborsRegressor(n_neighbors=5)
estimator.fit(x_train, y_train)

mse, mae, r2 = calculate_metrics(estimator, x_train, y_train)
print_metrics("Training metrics", mse, mae, r2)

mse, mae, r2 = calculate_metrics(estimator, x_test, y_test)
print_metrics("Testing metrics", mse, mae, r2)

