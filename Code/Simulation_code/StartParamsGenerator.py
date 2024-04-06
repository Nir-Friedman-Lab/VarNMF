from lib_code.EM_lib import *
from NMF_code.NMF import ExtendedNMF


class StartParamsGenerator:
    def __init__(self, params_dir, hyper_params=HyperParams(), dims=None, data=None):
        """
        :param params_dir: String dir of data_obj
        :param hyper_params: HyperParams
        :param dims: Dims(N:N, M:M, K:K)
        """
        self.dims = dims
        if dims: self.N, self.M, self.K = dims

        self.start = Params()
        self.hyper = hyper_params

        self.params_dir = params_dir
        self.params_obj_path = self.params_dir / 'params_obj.pkl'

        self.data = data

    @staticmethod
    def load_params(params_path, param_name='params_obj'):
        with (params_path/f'{param_name}.pkl').open('rb') as input:
            params = pickle.load(input)
        return params

    def dump_params(self, params_path, param_name='params_obj'):
        with (params_path/f'{param_name}.pkl').open('wb') as output:
            pickle.dump(self, output)

    def _load_start_params(self):
        try:
            with self.params_obj_path.open('rb') as input:
                param_obj = pickle.load(input)
        except ModuleNotFoundError:  # fix for unpickling old files (before split to packages) TODO
            with self.params_obj_path.open('rb') as input:
                param_obj = renamed_load(input)

        if self.dims is None: self.dims = param_obj.dims
        self.N, self.M, self.K = self.dims

        self.start = Params()
        if param_obj.start.a is not None:
            self.start.a = param_obj.start.a[:self.M, :self.K]
            self.start.b = param_obj.start.b[:self.M, :self.K]
        if param_obj.start.W is not None:
            self.start.W = param_obj.start.W[:self.N, :self.K]
        if param_obj.start.H is not None:
            self.start.H = param_obj.start.H[:self.M, :self.K]
        elif param_obj.start.H is None and param_obj.start.a is not None:
            self.start.H = self.start.a / self.start.b

        self.hyper = param_obj.hyper
        if self.hyper.H is not None: self.hyper.H = self.hyper.H[:self.M, :self.K]
        if self.hyper.W is not None: self.hyper.W = self.hyper.W[:self.N, :self.K]
        if type(self.hyper.bg) != int: self.hyper.bg = self.hyper.bg[:self.N]

        if param_obj.data is not None: self.data = param_obj.data[:self.N, :self.M]

    def _generate_start_params(self):
        pass

    def _save_start_params(self):
        self.start.save_csv(self.params_dir, prefix='start_')

    def _dump_params(self):
        with self.params_obj_path.open('wb') as output:
            pickle.dump(self, output)

    def generate(self):
        if not self.params_dir.exists():
            self.params_dir.mkdir(parents=True)
        if not self.params_obj_path.exists():
            print('Generating params in ' + str(self.params_dir))
            self._generate_start_params()
            self._save_start_params()
            self._dump_params()
        else:
            self._load_start_params()
            self._save_start_params()
            self._dump_params()
        return self


class InitStartParamsGenerator(StartParamsGenerator):
    def __init__(self, params_dir, hyper_params=HyperParams(), dims=None, data=None):
        """
        :param params_dir: String dir of data_obj
        :param hyper_params: HyperParams(W:start_W, a:start_a, b:start_b)
        :param dims: Dims(N:N, M:M, K:K)
        """
        super().__init__(params_dir, hyper_params=hyper_params, dims=dims, data=data)

    def _generate_start_params(self):
        if self.hyper.W is None:
            if self.hyper.H is not None:  # do NNR
                temp = NNRStartParamsGenerator(self.params_dir, hyper_params=self.hyper,  dims=self.dims, data=self.data)
                temp._generate_start_params()
                W = temp.start.W
            else: # do NMF
                temp = NMFStartParamsGenerator(self.params_dir, hyper_params=self.hyper, dims=self.dims, data=self.data)
                temp._generate_start_params()
                W = temp.start.W
        else: W = self.hyper.W
        if self.hyper.a is None:
            if self.hyper.H is not None:
                # var = 1
                var = (self.hyper.H.T ** 2)  # TODO
                a = (self.hyper.H.T ** 2) / var
                a[a < EPS] = EPS  # so a would be > 0
                b = self.hyper.H.T / var
                b[b < EPS] = EPS  # so b would be > 0
            else:
                assert False
        else:
            a, b = self.hyper.a, self.hyper.b
        self.start = Params(W=W, a=a, b=b, H=self.hyper.H.T if self.hyper.H is not None else None) # W.shape = NxK, a,b.shape = MxK


class NMFStartParamsGenerator(StartParamsGenerator):
    def __init__(self, params_dir, hyper_params=None, dims=None, data=None):
        """
        :param params_dir: String dir of data_obj
        :param hyper_params: HyperParams(init_NMF:init_NMF='random', beta:beta='KL', bg:bg)
        :param dims: Dims(N:N, M:M, K:K)
        :param data: np.array(N,M)
        """
        super().__init__(params_dir, hyper_params=hyper_params, dims=dims, data=data)

    def _generate_start_params(self):
        W, mu = ExtendedNMF(self.dims, init=self.hyper.init_NMF, beta=self.hyper.beta, solver=self.hyper.solver).fit(
            self.data, S=self.hyper.bg
        )
        # W, mu, _ = ExtendedNMF(self.dims, init=self.hyper.init_NMF, beta=self.hyper.beta, solver=self.hyper.solver).fit(
        #     self.data, S=self.hyper.bg, return_ll=True  todo
        # )
        W[W < EPS] = EPS  # so W would be > 0
        mu[mu < EPS] = EPS  # so W would be > 0
        self.start.W = W
        # self.start.H = mu.T
        # var = np.array([[np.var(self.data[:, j]) for j in range(self.dims.M)]] * self.dims.K).T.clip(EPS, 100)
        # var = 1  # TODO
        var = (mu.T ** 2)  # TODO
        # var[var < .1] = .1

        a = (mu.T ** 2) / var
        a[a < EPS] = EPS  # so a would be > 0
        self.start.a = a
        b = mu.T / var
        b[b < np.sqrt(EPS)] = np.sqrt(EPS)  # so b would be > 0
        self.start.b = b
        self.start.H = a/b


class NNRStartParamsGenerator(StartParamsGenerator):
    def __init__(self, params_dir, hyper_params=None, dims=None, data=None):
        """
        :param params_dir: String dir of data_obj
        :param hyper_params: HyperParams(H)
        :param dims: Dims(N:N, M:M, K:K)
        :param data: np.array(N,M)
        """
        super().__init__(params_dir, hyper_params=hyper_params, dims=dims, data=data)

    def _generate_start_params(self):
        # W = np.array([nnls(self.hyper.H, self.data[i])[0] for i in range(self.N)]) # this is for normal noise and not poisson
        W, _ = ExtendedNMF(self.dims, init=self.hyper.init_NMF, beta=self.hyper.beta, solver=self.hyper.solver).fit_W(
            self.data, H=self.hyper.H.mean(axis=0).T if len(self.hyper.H.shape) == 3 else self.hyper.H.T, S=self.hyper.bg
        )
        W[W < EPS] = EPS  # so W would be > 0
        self.start.W = W





