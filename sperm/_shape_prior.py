# Author: Xiang CHEN <xiangchen.ai@outlook.com>
# License: Apache Software License

PRIOR_TYPE_LIST = [
    'nonnegative', 'nonpositive',
    'increasing', 'decreasing',
    'Lipschitz',
    'quasi-convex', 'quasi-concave',
    'convex', 'concave',
]

BASE_MODEL_LIST = [
    'linear',
    'mlp',
]

class ShapePrior:
    """
    class for Shape Priors. For each base models:
    - 'linear': prior_type should be one of: 'increasing', 'decreasing', 'Lipschitz'.

    Parameters
    ----------
    base_model: str
        Base model name. The string should be one of the elements in BASE_MODEL_LIST.
    prior_list: list
        List of strings following one of the following format:
        - dim:prior_type, where dim is an integer indicating the dimension of the input feature, and prior_type is one of the following: 'nonnegative', 'nonpositive', 'increasing', 'decreasing', 'quasi-convex', 'quasi-concave', 'convex', 'concave'.
        - dim:prior_type:arg, where dim is an integer indicating the dimension of the input feature, prior_type is 'Lipschitz', and arg is a positive number indicating the Lipschitz constant.

    Attributes
    ----------
    base_model: str
        Base model name.
    prior_list: list
        List of tuples containing shape prior information.
    """
    def __init__(self, base_model, prior_list):
        assert base_model in BASE_MODEL_LIST, "Invalid base_model input: %s"%base_model
        self.base_model = base_model

        self.prior_list = []
        # parsing and checking prior_list
        for p in prior_list:
            metas = p.split(':')
            assert (len(metas)>=2 and (metas[1] in PRIOR_TYPE_LIST)), "Invalid shape_prior input: %s"%p
            if metas[1] in ['nonnegative', 'nonpositive',
                            'increasing', 'decreasing',
                            'quasi-convex', 'quasi-concave',
                            'convex', 'concave']:
                if len(metas)!=2:
                    raise ValueError("Invalid shape_prior input: %s"%p)
                self.prior_list.append((int(metas[0]), metas[1]))
            elif metas[1] in ['Lipschitz']:
                if len(metas)!=3:
                    raise ValueError("Invalid shape_prior input: %s"%p)
                if float(metas[2])<=0:
                    raise ValueError("Lipschitz constant cannot be negative: %s"%p)
                self.prior_list.append((int(metas[0]), metas[1], float(metas[2])))

        # base_model-specific prior validity check
        if self.base_model=='linear': # including LinearRegression and Ridge
            for p in self.prior_list:
                if not p[1] in ['increasing', 'decreasing', 'Lipschitz']:
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
            if p[1] in ['nonnegative', 'nonpositive',
                        'increasing', 'decreasing',
                        'quasi-convex', 'quasi-concave',
                        'convex', 'concave']:
                str_prior_list.append('[dim %d, %s]'%(p[0], p[1]))
            elif p[1] in ['Lipschitz']:
                str_prior_list.append('[dim %d, %s, constant: %d]'%(p[0], p[1], p[2]))
        return 'shape_prior(' + ', '.join(str_prior_list) + ')'

__all__ = ['ShapePrior']
