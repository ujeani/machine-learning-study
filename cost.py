# Get cost functon for Linear regression


# Cost function for Hypothesis h(x) = Wx+b
# tdata : training data

def getcost(tdata, w, b):
    cost = 0
    iter = 0

    for (x, y) in tdata :
        h = w*x+b
        cost += (h-y)**2
        iter += 1

    cost /= (2.0*float(iter))
    return cost

if __name__ == "__main__":
    # Training Data set (x, y)
    training_data = [(1.0, 2.0), (2.0, 4.0), (3.0, 5.0)]

    minc = 100
    minw = 0
    minb = 0
    for witer in range(0, 1001) :
        for biter in range(0, 1001) :
            w = witer/100.0
            b = biter/100.0
            c = getcost(training_data, w, b)
            # print("cost(w={0}, b={1}) = {2}".format(w, b, c))
            if(minc > c) :
                minc = c
                minw = w
                minb = b
    print("Minimized hypothesis h(x) = {0}x+{1} with cost={2}".format(minw, minb, minc))