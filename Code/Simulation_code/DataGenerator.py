from lib_code.EM_lib import *
from Simulation_code import gamma_model


class DataGenerator:
    def __init__(self, data_dir, true_params=Params(), hyper_params=HyperParams(), dims=None, prefix=''):
        """
        :param data_dir: String dir of data_obj
        :param true_params: Params()
        :param hyper_params: HyperParams(bg:bg=0)
        :param dims: Dims(N:N, M:M, K:K)
        """
        self.dims = dims
        if dims: self.N, self.M, self.K = dims

        self.data = None
        self.true = true_params
        self.hyper = hyper_params

        self.data_dir = data_dir
        self.data_obj_path = self.data_dir/(prefix + 'data_obj.pkl')

    def _load_true_params(self):
        try:
            with self.data_obj_path.open('rb') as input:
                data_obj = pickle.load(input)
        except ModuleNotFoundError:  # fix for unpickling old files (before split to packages) TODO
            with self.data_obj_path.open('rb') as input:
                data_obj = renamed_load(input)

        if self.dims is None: self.dims = data_obj.dims
        self.N, self.M, self.K = self.dims

        self.data = data_obj.data[:self.N, :self.M]

        self.true = Params()
        if data_obj.true.a is not None:
            self.true.a = data_obj.true.a[:self.M, :self.K]
            self.true.b = data_obj.true.b[:self.M, :self.K]
        if data_obj.true.H is not None:
            self.true.H = data_obj.true.H
        if data_obj.true.lam is not None and type(data_obj.true.lam) != int:
            self.true.lam = data_obj.true.lam[:self.N]
        if data_obj.true.W is not None:
            self.true.W = data_obj.true.W[:self.N, :self.K]
            if not np.all(np.isclose(self.true.W.sum(axis=1), 1)): self.true.lam = 1  # TODO!!!
        self.true.var = data_obj.true.var

        if self.hyper.W is not None: self.hyper.W = self.hyper.W[:self.N, :self.K]
        if type(self.hyper.bg) != int: self.hyper.bg = self.hyper.bg[:self.N, :self.M]

    def _generate_true_params(self):
        pass

    def _save_true_params(self):
        pd.DataFrame.to_csv(pd.DataFrame(self.data), self.data_dir/'data.csv', index=False)
        self.true.save_csv(self.data_dir, 'true_')

    def _dump_data(self):
        with self.data_obj_path.open('wb') as output:
            pickle.dump(self, output)

    def generate(self):
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)
        if not self.data_obj_path.exists():
            print('Generating data in ' + str(self.data_dir))
            self._generate_true_params()
            self._save_true_params()
            self._dump_data()
        else:
            self._load_true_params()
            # self._save_true_params() # TODO
            # self._dump_data()
        return self

    def copy(self, other):
        self.data_dir = other.data_dir
        self.dims = Dims(N=other.dims.N, M=other.dims.M, K=other.dims.K)
        self.N, self.M, self.K = self.dims
        self.data = other.data
        self.true = Params().copy(other.true)
        self.hyper = HyperParams().copy(other.hyper)
        self.data_dir = other.data_dir
        self.data_obj_path = other.data_obj_path
        return self


class InitDataGenerator(DataGenerator):
    def __init__(self, data_dir, true_params=Params(), hyper_params=HyperParams(), dims=None, prefix=''):
        """
        :param data_dir: String dir of data_obj
        :param true_params: Params(lam:lam=1, W:true_W, a:true_a, b:true_b)
        :param hyper_params: HyperParams(bg:bg=0)
        :param dims: Dims(N:N, M:M, K:K)
        """
        assert true_params.W is not None, "InitDataGenerator should be initialized with W"
        assert true_params.a is not None, "InitDataGenerator should be initialized with a"
        assert true_params.b is not None, "InitDataGenerator should be initialized with b"
        super().__init__(data_dir, true_params=true_params, hyper_params=hyper_params, dims=dims, prefix=prefix)

    def _generate_true_params(self):
        self.true.H = gamma_model.sample_gamma(self.true.a, self.true.b, self.dims)
        self.data, self.Y = gamma_model.sample_sum_poi(self.true.W, self.true.H, self.dims,
                                                       lam=self.true.lam, bg=self.hyper.bg)


class AtlasDataGenerator(DataGenerator):
    def __init__(self, data_dir, true_params=Params(), hyper_params=HyperParams(), dims=None, prefix=''):
        """
        :param data_dir: String dir of data_obj
        :param true_params: Params(lam:lam=1, W:true_W, H:atlas)
        :param hyper_params: HyperParams(bg:bg=0)
        :param dims: Dims(N:N, M:M, K:K)
        """
        # assert true_params.H is not None, "AtlasDataGenerator should be initialized with H"
        super().__init__(data_dir, true_params=true_params, hyper_params=hyper_params, dims=dims, prefix=prefix)

    def _save_true_params(self):
        pd.DataFrame.to_csv(pd.DataFrame(self.data), self.data_dir/'data.csv', index=False)
        self.true.save_csv(self.data_dir, 'true_')

    def _generate_true_params(self):
        if self.true.W is None:
            self.true.W = np.random.dirichlet([1] * self.K, size=self.N)
        self.true.W = self.true.lam * self.true.W
        self.true.lam = 1
        # a = mu^2/sigma^2 = atlas^2/(var * atlas) = atlas/var, a dims = M x K
        # b = mu/sigma^2 = atlas/(var * atlas) = 1/var, b dims = M, K
        if self.true.var > 0:
            self.true.a = self.true.H / (self.true.var + EPS)
            self.true.a[self.true.a < EPS] = EPS  # so a would be > 0
            self.true.b = np.zeros((self.M, self.K)) + 1 / (self.true.var + EPS)
            self.true.H = gamma_model.sample_gamma(self.true.a, self.true.b, self.dims)
        else:
            self.true.H = np.broadcast_to(self.true.H[None, :, :], self.dims)
        self.data, self.Y = gamma_model.sample_sum_poi(self.true.W, self.true.H, self.dims,
                                                       lam=1, bg=self.hyper.bg)

    def union(self, other):
        assert self.K == other.K and self.M == other.M, "Cannot union with different M, K"

        self.N = self.N + other.N
        self.dims.N = self.N
        self.data = np.concatenate([self.data, other.data], axis=0)
        self.true = Params(W=np.concatenate([self.true.W, other.true.W], axis=0))
        return self


class DataInitDataGenerator(DataGenerator):
    def __init__(self, data_dir, true_params=Params(), hyper_params=HyperParams(), dims=None, prefix=''):
        """
        :param data_dir: String dir of data_obj
        :param true_params: Params(lam:lam=1, W:true_W, a:true_a, b:true_b)
        :param hyper_params: HyperParams(bg:bg=0, data=data)
        :param dims: Dims(N:N, M:M, K:K)
        """
        super().__init__(data_dir, true_params=true_params, hyper_params=hyper_params, dims=dims, prefix=prefix)
        self.data = hyper_params.data

