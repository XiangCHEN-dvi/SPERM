PRIOR_TYPE_LIST = [
    'nonnegative', 'nonpositive',
    'increasing', 'decreasing',
    'Lipschitz',
    'quasi-convex', 'quasi-concave',
    'convex', 'concave',
]

class ShapePrior:
    def __init__(self, base_model, prior_list):
        self.base_model = base_model
        self.prior_list = []

        # parsing and checking prior_list
        for p in prior_list:
            metas = p.split(':')
            if metas[1] not in PRIOR_TYPE_LIST:
                raise ValueError("Invalid shape_prior input: %s"%p)
            elif metas[1] in ['nonnegative','nonpositive',
                              'increasing','decreasing',
                              'quasi-convex','quasi-concave',
                              'convex','concave']:
                if len(metas)!=2:
                    raise ValueError("Invalid shape_prior input: %s"%p)
                self.prior_list.append([int(metas[0]), metas[1]])
            elif metas[1] in ['Lipschitz']:
                if len(metas)!=3:
                    raise ValueError("Invalid shape_prior input: %s"%p)
                if int(metas[2])<0:
                    raise ValueError("Lipschitz constant cannot be negative: %s"%p)
                self.prior_list.append([int(metas[0]), metas[1], int(metas[2])])
    
        # base_model specific prior validity check
        if self.base_model=='linear': # including LinearRegression and Ridge
            for p in self.prior_list:
                if not p[1] in ['increasing','decreasing','Lipschitz']:
                    raise TypeError("Prior %s not supported for linear models"%p[1])
        elif self.base_model=='polynomial':
            pass
        elif self.base_model=='tree':
            pass
        elif self.base_model=='mlp':
            pass

    def __str__(self):
        str_prior_list = []
        for p in self.prior_list:
            if p[1] in ['nonnegative','nonpositive',
                        'increasing','decreasing',
                        'quasi-convex','quasi-concave',
                        'convex','concave']:
                str_prior_list.append('[dim %d, %s]'%(p[0], p[1]))
            elif p[1] in ['Lipschitz']:
                str_prior_list.append('[dim %d, %s, constant: %d]'%(p[0], p[1], p[2]))
        return 'shape_prior(' + ', '.join(str_prior_list) + ')'

__all__ = ['ShapePrior']
