import cost

# Training Data set (x, y)
training_data = [(1.0, 2.0), (2.0, 4.0), (3.0, 5.0)]

minc = 100
minw = 0
minb = 0
for witer in range(0, 1001):
    for biter in range(0, 1001):
        w = witer / 100.0
        b = biter / 100.0
        c = cost.getcost(training_data, w, b)
        # print("cost(w={0}, b={1}) = {2}".format(w, b, c))
        if (minc > c):
            minc = c
            minw = w
            minb = b

print("Minimized hypothesis h(x) = {0}x+{1} with cost={2}".format(minw, minb, minc))