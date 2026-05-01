import matplotlib.pyplot as plt


def plot_forecast(model, forecast):
    fig = model.model.plot(forecast)
    plt.show()


def plot_components(model, forecast):
    fig = model.model.plot_components(forecast)
    plt.show()