import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(data_x, data_y):
    m = b = 0
    learning_rate = 0.001
    iterations = 10000

    n = len(data_x)
    
    for i in range(iterations):
        y_pred = m * data_x + b
    
        cost = (1 / n) * np.sum((data_y - y_pred) ** 2)
    
        md = -(2 / n) * np.sum(data_x * (data_y - y_pred))
        bd = -(2 / n) * np.sum(data_y - y_pred)
    
        m -= learning_rate * md
        b -= learning_rate * bd
    
        print("m {}, b {}, cost {}, iter {}".format(m, b, cost, i))

    return m, b


if __name__ == "__main__":
    np.random.seed(42)

    n = 50
    x = np.linspace(1, 50, n)
    y = 4 * x + 7 + np.random.randn(n) * 10

    m, b = gradient_descent(x, y)


    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="blue", label="Data points", alpha=0.7)
    plt.plot(x, m * x + b, color="red", linewidth=2, label=f"Fitted line: y={m:.2f}x+{b:.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression with Gradient Descent")
    plt.legend()
    plt.grid(True, linestyle="--")
    plt.show()
