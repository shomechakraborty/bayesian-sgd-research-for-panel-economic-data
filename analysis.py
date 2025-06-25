from neural_network_class import NeuralNetworkModel
import bayesian_optimization as bo
from sklearn.neural_network import MLPRegressor
import math, statistics as stats, copy, random

"""This function runs the Bayesian Optimized Stochastic Gradient Descent (SGD) regression neural network along with a standard
MLP Regressor neural network Economic testing data for benchmark testing and reports each model's RMSE for 50 trials, as well as
the total weighted feature influences of each trial Bayesian Optimizzed SGD model."""
def main():
    input_data_file_names = ["business_applications_processed.csv", "manufacturing_and_trade_inventories_and_sales_processed.csv", "construction_spending_rate_processed.csv", "advance_retail_and_food_sales_processed.csv", "new_manufacturer_shipments_inventories_and_orders_processed.csv", "international_goods_and_services_trade_processed.csv"]
    target_data_file_name = "disposable_personal_income_processed.csv"
    feed_input_data, testing_input_data, feed_target_data, testing_target_data = process_data(input_data_file_names, target_data_file_name, 2005, 2025, 0.8)
    feed_data_reference = {
    "input_data": feed_input_data,
    "target_data": feed_target_data
    }
    hyperparameters = {
    "model_size": (5, True, 2, 7, 6),
    "neuron_size_base": (2, True, 2, 4, 3),
    "training_epochs": (75, True, 50, 350, 301),
    "training_data_proportion": (0.75, False, 0.70, 0.90, 1000),
    "delta": (1.0, False, 0.001, 1, 1000),
    "learning_rate": (0.01, False, 0.0001, 0.1, 1000),
    "learning_rate_decay_rate": (0.001, False, 0.00001, 0.01, 1000),
    "momentum_factor": (0.9, False, 0.001, 1, 1000),
    "max_norm_benchmark": (90, False, 80, 100, 1000),
    "l2": (0.01, False, 0.0001, 0.1, 1000)
    }
    cost_attribute_reference = {
    "cost": "get_average_validation_cost_values_over_epochs()[model.get_minimum_cost_index()]"
    }
    print("Testing Target Data Mean - " + str(stats.mean(testing_target_data)))
    model_rmse_across_trials = []
    mlp_model_rmse_across_trials = []
    model_total_weighted_feature_influences_across_trials = []
    for i in range(50):
        success = False
        while not success:
            try:
                optimal_model_hyperparameters = bo.optimize_neural_network_hyperparameters(NeuralNetworkModel, feed_data_reference, hyperparameters, cost_attribute_reference, 0, False)
                model = NeuralNetworkModel(**feed_data_reference, **optimal_model_hyperparameters)
                mlp_model = MLPRegressor()
                mlp_model.fit(feed_input_data, feed_target_data)
                model_testing_predictions = model.run_model(testing_input_data)
                mlp_model_testing_predictions = mlp_model.predict(testing_input_data)
                model_ses = []
                mlp_model_ses = []
                for j in range(len(testing_target_data)):
                    model_ses.append(pow(testing_target_data[j] - model_testing_predictions[j], 2))
                    mlp_model_ses.append(pow(testing_target_data[j] - mlp_model_testing_predictions[j], 2))
                model_mse = stats.mean(model_ses)
                mlp_model_mse = stats.mean(mlp_model_ses)
                model_rmse = math.sqrt(model_mse)
                mlp_model_rmse = math.sqrt(mlp_model_mse)
                model_total_weighted_feature_influences = calculate_total_weighted_feature_influences(model)
                model_rmse_across_trials.append(model_rmse)
                mlp_model_rmse_across_trials.append(mlp_model_rmse)
                model_total_weighted_feature_influences_across_trials.append(model_total_weighted_feature_influences)
                success = True
            except Exception:
                continue
        print("Trial " + str(i + 1) + " Results")
        print("Model RMSE - " + str(model_rmse))
        print("MLP Model RMSE - " + str(mlp_model_rmse))
        print("Model Total Weighted Feature Influences - " + str(model_total_weighted_feature_influences))
    print("*****")
    print("Model Results")
    print(model_rmse_across_trials)
    print("*****")
    print("MLP Model Results")
    print(mlp_model_rmse_across_trials)
    print("*****")
    print("Model Total Weighted Feature Influences Results")
    print(model_total_weighted_feature_influences_across_trials)

"""This model processes Economic datasets and returns the respective feed and testing input and target datasets to be
used for testing between the Bayesian Optimized SGD and MLP Regressor neural networks."""
def process_data(input_data_file_names, target_data_file_name, start_year, end_year, feed_data_proportion):
    random.seed(42)
    input_data = []
    target_data = []
    months = ((end_year - start_year) * 12) + 1
    for i in range(months):
        input_data.append([])
    for i in range(len(input_data_file_names)):
        file = open(input_data_file_names[i])
        j = 0
        for line in file:
            data_line = line.strip().split(",")
            if data_line[0] == "" or data_line[0] == "Period":
                continue
            year = int(data_line[0][-4:])
            month = data_line[0][:-5]
            if year < start_year or year > end_year or (year == end_year and month != "Jan"):
                continue
            value = float(data_line[1])
            input_data[j].append(value)
            j += 1
    file = open(target_data_file_name)
    j = 0
    for line in file:
        data_line = line.strip().split(",")
        if data_line[0] == "observation_date":
            continue
        year = int(data_line[0][:4])
        month = int(data_line[0][5:7])
        if year < start_year or year > end_year or (year == start_year and month == 1) or (year == end_year and month != 1 and month != 2):
            continue
        target_data.append(float(data_line[1]))
        j += 1
    indices = random.sample(range(len(input_data)), len(input_data))
    shuffled_input_data = [input_data[i] for i in indices]
    shuffled_target_data = [target_data[i] for i in indices]
    feed_input_data = shuffled_input_data[:round(len(shuffled_input_data) * feed_data_proportion)]
    testing_input_data = shuffled_input_data[round(len(shuffled_input_data) * feed_data_proportion):]
    feed_target_data = shuffled_target_data[:round(len(shuffled_target_data) * feed_data_proportion)]
    testing_target_data = shuffled_target_data[round(len(shuffled_target_data) * feed_data_proportion):]
    return feed_input_data, testing_input_data, feed_target_data, testing_target_data

"""This function determines the total weighted feature weights of a given Bayesian Optimized SGD model using a
backward weighting propagation method."""
def calculate_total_weighted_feature_influences(model):
    parameters = model.get_parameters()
    total_weighted_input_influences_across_layers = []
    for i in range(len(parameters)):
        total_weighted_input_influences_across_layers.append([])
    for i in reversed(range(len(parameters))):
        layer_total_weighted_input_influences_across_neurons = []
        for j in range(len(parameters[i])):
            if i != len(parameters) - 1:
                neuron_total_weighted_input_influences = [parameters[i][j][0][k] * total_weighted_input_influences_across_layers[i + 1][j] for k in range(len(parameters[i][j][0]))]
            else:
                neuron_total_weighted_input_influences = [parameters[i][j][0][k] for k in range(len(parameters[i][j][0]))]
            layer_total_weighted_input_influences_across_neurons.append(neuron_total_weighted_input_influences)
        layer_total_weighted_input_influences = [sum([layer_total_weighted_input_influences_across_neurons[k][j] for k in range(len(layer_total_weighted_input_influences_across_neurons))]) for j in range(len(layer_total_weighted_input_influences_across_neurons[0]))]
        total_weighted_input_influences_across_layers[i] = layer_total_weighted_input_influences
    return total_weighted_input_influences_across_layers[0]

main()
