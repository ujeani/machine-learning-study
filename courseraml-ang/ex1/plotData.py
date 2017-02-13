import matplotlib.pyplot as plt

def plot(x, y):
    plt.plot(x, y, 'rx')
    plt.title('MarkerSize')
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.show()