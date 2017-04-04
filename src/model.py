
def FactorizationMachine(in_dim, factor_dim, cate_dim)
    # layer
    data = mx.sym.Variable('data')
    net = mx.sym.Embedding(data=data, input_dim=in_dim, output_dim=factor_dim)
    net = mx.symbol.Reshape(data=net, shape=(-1, cate_dim, factor_dim))
    net = mx.symbol.sum(data=net, axis=1)
    net = mx.symbol.square(data=net)
    net = mx.symbol.sum(data=net, axis=1)
    net = mx.symbol.Reshape(data=net, shape=(-1, 1))

#linear
    net0 = mx.symbol.Embedding(data = data, input_dim = in_dims , output_dim = 1)
    net0 = mx.symbol.sum(data=net0, axis=1)
    net += net0

    net = mx.symbol.LogisticRegressionOutput(data=net, name='softmax')
