import numpy as np
import matplotlib.pyplot as plt

# Class to gain access to iris dataset
class IrisWrapper:
    def __init__(self, data_path: str) -> None:
        self.data = []
        self.feed_data(data_path)


    def feed_data(self, path: str):
        with open(path, "r") as file:
            rows = file.readlines()
            self.data = []
            for row in rows:
                if row.strip() == "": continue
                self.data.append(row.replace("/n", "").split(","))
        self.format_data()

    def format_data(self):
        for entry in self.data:
            for i in range(len(entry) - 1):
                entry[i] = float(entry[i])

class GradientDescentOneDim(IrisWrapper):   
    def error_gradient(self, w: list, x_col_idx: int, y_col_idx: int):
        columns = list(zip(*self.data))
        x = columns[x_col_idx]
        y = columns[y_col_idx]
        res = 0
        for i in range(len(x)):
            res += -2*y[i]*x[i]+2*w*(x[i])**2
        return res * 1/len(x)
    
    def error_function(self, w: list, x_col_idx: int, y_col_idx: int):
        columns = list(zip(*self.data))
        x = columns[x_col_idx]
        y = columns[y_col_idx]
        res = 0
        for i in range(len(x)):
            res += (y[i]-w*x[i])**2
        return res * 1/len(x)

    def plot_error_derivation(self, x_col_idx: int, y_col_idx: int):
        x_lin_space = list(range(-100, 100))
        y_values = []
        for x_value in x_lin_space:
            y_values.append(self.error_gradient(x_value, x_col_idx, y_col_idx))
        plt.plot(x_lin_space, y_values) 
        plt.show()

    def plot_error_function(self, x_col_idx: int, y_col_idx: int):
        x_lin_space = np.arange(-4, 4, 0.1).tolist()
        y_values = []
        for x_value in x_lin_space:
            y_values.append(self.error_function(x_value, x_col_idx, y_col_idx))
        plt.plot(x_lin_space, y_values, linewidth=2.0, color="red") 

    def gradient_descent(self, x_col_idx: int, y_col_idx: int, start_x: float, learning_rate: float, exit_gradient: float):
        self.plot_error_function(x_col_idx, y_col_idx)
        current_x = start_x
        i = 0
        gradient = self.error_gradient(current_x, x_col_idx, y_col_idx)
        while abs(gradient) > exit_gradient:
            i += 1
            print(gradient)
            gradient = self.error_gradient(current_x, x_col_idx, y_col_idx)
            y_0 = self.error_function(current_x, x_col_idx, y_col_idx)
            x_0 = current_x
            # adjust weight
            current_x -= learning_rate * gradient

            # Plotting
            x_lin_space = [current_x, current_x + 5]
            y_values = []
            for x in x_lin_space:
                y_values.append(y_0 + self.error_gradient(x_0, x_col_idx, y_col_idx) * (x - x_0))
            c = np.random.rand(3)
            s = 30
            if abs(gradient) <= exit_gradient:
                c = 'red'
                s = 50
            if i % 30 == 0 or abs(gradient) <= exit_gradient:
                plt.plot(x_lin_space, y_values, c='gray', zorder=0)
                plt.scatter([x_0], [y_0], c=c, s=s, zorder=10)
        plt.show()
        return current_x
    
    def plot_regression_function(self, x_col_idx: int, y_col_idx: int, w: list):
        columns = list(zip(*self.data))
        x = columns[x_col_idx]
        y = columns[y_col_idx]

        x_lin_space = np.linspace(0, 10, 100)
        y_hat = w * x_lin_space
        plt.scatter(x, y, marker='x')
        plt.plot(x_lin_space, y_hat, color='r')
        plt.show()


from mpl_toolkits.mplot3d import Axes3D  

class GradientDescentMultiDim(IrisWrapper):
    def error_function(self, w: list, x_col_idx: int, y_col_idx: int):
        columns = list(zip(*self.data))
        x = columns[x_col_idx]
        y = np.asarray(columns[y_col_idx])
        X = np.asarray([x, np.ones(len(x))]).T
        beta = np.asarray(w).T
        return (y - X@beta).T @ (y - X@beta)
    
    def error_gradient(self, w: list, x_col_idx: int, y_col_idx: int):
        columns = list(zip(*self.data))
        x = columns[x_col_idx]
        y = np.asarray(columns[y_col_idx])
        X = np.asarray([x, np.ones(len(x))]).T
        beta = np.asarray(w).T
        return -2*X.T@y+2*X.T@X@beta
    
    def plot_error_function(self, x_col_idx: int, y_col_idx: int):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = y = np.arange(-4, 4, 0.05)
        X, Y = np.meshgrid(x, y)
        X_flat = np.ravel(X).tolist()
        Y_flat = np.ravel(Y).tolist()
        zs = np.array([])
        for i in range(len(X_flat)):
            zs = np.append(zs, [self.error_function([X_flat[i], Y_flat[i]], x_col_idx, y_col_idx)])
        Z = zs.reshape(X.shape)
        ax.plot_surface(X, Y, Z, zorder=1, cmap='plasma', alpha=0.6,)
        return ax

    def gradient_descent_3D(self, x_col_idx: int, y_col_idx: int, starting_weights: list, learning_rate: float, exit_gradient: float, iterations: int):
        current_weights = np.asarray(starting_weights)
        current_gradient = self.error_gradient(starting_weights, x_col_idx, y_col_idx)
        ax = self.plot_error_function(x_col_idx, y_col_idx)
        i = 0
        while sum(current_gradient) > exit_gradient and i <= iterations:
            ax.scatter([current_weights[0]], [current_weights[1]], [self.error_function(current_weights.tolist(), x_col_idx, y_col_idx)], c='red' if i == iterations  else 'black', s=40 if i == iterations else 10, zorder=3)
            i += 1
            current_gradient = self.error_gradient(current_weights, x_col_idx, y_col_idx)
            current_weights = current_weights - current_gradient * learning_rate
        plt.title(f'Error Function | Final Weights: {current_weights}')
        plt.show()
        return current_weights
    
            
    def plot_regression_function(self, x_col_idx: int, y_col_idx: int, w):
        columns = list(zip(*self.data))
        x = columns[x_col_idx]
        y = columns[y_col_idx]

        x_lin_space = np.linspace(0, 10, 100)
        y_hat = w[0] * x_lin_space + w[1]
        plt.scatter(x, y, marker='x')
        plt.plot(x_lin_space, y_hat, color='r')
        plt.ylim(ymin=0)
        plt.title(f'Regression Function | slope: {round(w[0], 3)} y intercept: {round(w[1], 3)}')
        plt.show()


instance = GradientDescentMultiDim('iris.csv')
weights = instance.gradient_descent_3D(0, 1, [2, 2], 0.00004, 5, 10)
instance.plot_regression_function(0, 1, weights)