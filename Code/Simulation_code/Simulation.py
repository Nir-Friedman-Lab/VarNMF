from EM_code.EM import EM
from EM_code.EMk import EMkH, EMkAB, EMkW
from EM_code.batchEM import batchEM, batchEMkAB, batchEMkW
from Simulation_code.StartParamsGenerator import *
from NMF_code.ExtendedNMFasEM import ExtendedNMFasEM, ExtendedNMFasEMkH


class Simulation:
    def __init__(self, output_sim_dir, EM_type=EM, sub_dir_name=''):

        self.EM_type = EM_type
        self.sim_dir = output_sim_dir/(sub_dir_name + "/" if len(sub_dir_name) > 0 else "")
        self.eps = np.inf
        self.max_itr = 0
        self.ll_scores = []
        self.prog = ParamsProg()  # results of all iterations
        self.hat = None  # results of last iteration (permuted?)
        self.start = Params()  # start parameters
        self.dims = None
        self.data = None
        self.true = None  # true parameters
        self.hyper = None  # parameters that shouldn't change during the simulation
        self.posterior_H = None
        self.itr_step_calc_ll, self.itr_step_save = 2, 15

    def __repr__(self):
        return (f"{type(self).__name__}\n" +
                f"\tsim dir:\t{Path('/'.join(self.sim_dir.parts[2:]))}\n" +
                f"\tcur eps:\t{self.eps:.10f}\n" +
                f"\tcur itr:\t{self.max_itr}")

    def __copy__(self, other):
        self.eps = other.eps
        self.max_itr = other.max_itr
        self.EM_type = other.EM_type
        self.ll_scores = other.ll_scores
        self.prog = other.prog
        self.hat = other.hat
        self.start = other.start
        self.dims = other.dims
        self.data = other.data
        self.true = other.true
        self.hyper = other.hyper

    def _load_simulation(self):
        if (self.sim_dir/'simulation_obj.pkl').exists():
            try:
                with (self.sim_dir/'simulation_obj.pkl').open('rb') as input:
                    simulator = pickle.load(input)
                    self.__copy__(simulator)
                return True
            except ModuleNotFoundError:  # fix for unpickling old files (before split to packages)
                with (self.sim_dir / 'simulation_obj.pkl').open('rb') as input:
                    simulator = renamed_load(input)
                    self.__copy__(simulator)
                return True
            except Exception as e:
                print('could not load simulation_obj', e)
                return None
        return False

    def setup(self, param_obj=None, data_obj=None):
        if not self.sim_dir.exists():
            self.sim_dir.mkdir(parents=True)

        load = self._load_simulation()

        if not load:
            assert data_obj, "Couldn't find a data object, should be supplied"
            assert param_obj, "Couldn't find a param object, should be supplied"

            # params from data_obj
            self.dims = data_obj.dims
            self.data = data_obj.data
            self.true = Params().copy(data_obj.true)  # true parameters
            self.hyper = HyperParams().copy(data_obj.hyper)  # parameters that shouldn't change during the simulation

            # params from param_obj
            self.start = Params().copy(param_obj.start)
        return self

    @staticmethod
    def permute_res(pred, true, K):
        # find the permutation that get the closest pred (all permutation will be with the same ll). todo check
        C = np.array([[np.sum((pred.T[k] - true.T[t]) ** 2) for k in range(K)] for t in range(K)])
        perm = linear_sum_assignment(C)[1]
        return pred[:, perm]

    @staticmethod
    def print_em_itr(cur_itr, ll, last_diff):
        print(f"t: {cur_itr}\t ll: {ll:.5f}\t diff: {last_diff:.10f}", flush=True)

    def _save_results(self):
        print("Save", flush=True)
        pd.DataFrame.to_csv(pd.DataFrame(self.ll_scores), self.sim_dir/'ll_scores.csv', index=False)
        self.prog.save_csv(self.sim_dir, prefix='hat_')

    def _dump_simulation(self):
        print("Dump", flush=True)
        if len(self.ll_scores) >= self.itr_step_calc_ll:
            if self.itr_step_calc_ll == 1: self.eps = self.ll_scores[-1] - self.ll_scores[-2]
            else: self.eps = self.ll_scores[-1] - self.ll_scores[-self.itr_step_calc_ll]
            if self.eps == np.inf: self.eps = 0
        self.max_itr = len(self.ll_scores)
        with (self.sim_dir/'simulation_obj.pkl').open('wb') as output:
            pickle.dump(self, output)

    def get_em_obj(self):
        params = self.start if self.hat is None else self.hat
        return self.EM_type(self.data, params, self.hyper, self.dims)

    def _run_EM(self, em, eps, max_itr=0):
        cur_itr = self.max_itr
        ll = -np.inf
        last_diff = 0
        if cur_itr >= self.itr_step_calc_ll:
            if self.itr_step_calc_ll == 1: last_diff = self.ll_scores[-1] - self.ll_scores[-2]
            else: last_diff = self.ll_scores[-1] - self.ll_scores[-self.itr_step_calc_ll]
        if cur_itr >= 1: ll = self.ll_scores[-1]

        run_em_generator = em._fit_generator(
                cur_itr, last_diff, ll, eps, max_itr=max_itr, itr_step_calc_ll=self.itr_step_calc_ll
        )
        for cur_itr, ll, last_diff, params in run_em_generator:
            self.print_em_itr(cur_itr, ll, last_diff)

            self.ll_scores.append(ll)
            self.hat = params
            self.prog = self.prog + params

            if cur_itr % self.itr_step_save == 0:
                self._save_results()
                self._dump_simulation()

        return self.ll_scores, self.prog

    def _run_new_simulation(self, eps, max_itr):
        print("Running", flush=True)
        em = self.get_em_obj()
        self._run_EM(em, eps, max_itr)

    def _resume_simulation(self, eps, max_itr):
        print("Updating", flush=True)
        eps = eps if eps > 0 else self.eps
        em = self.get_em_obj()
        self._run_EM(em, eps, max_itr)

    def _if_cond_updating(self, eps, max_itr):
        return (0 < eps < self.eps) or (max_itr > 0 and max_itr > self.max_itr) or (max_itr == 0 and self.eps < -eps)

    @timer
    def run_simulation(self, eps=0, max_itr=0):
        print(f'Run with eps={eps}, max_itr={max_itr}', flush=True)
        flag_run = False
        flag_err = False
        try:
            if self.hat is None:  # new simulation
                flag_run = True
                self._run_new_simulation(eps, max_itr)
            else:  # resume
                if self._if_cond_updating(eps, max_itr):
                    flag_run = True
                    self._resume_simulation(eps, max_itr)
                else: print("Already updated", flush=True)
        except Exception as e:
            print('err', str(e), flush=True)
            flag_err = True
        finally:
            if flag_run:
                self._save_results()
                self._dump_simulation()
            print("Done", flush=True)
            if flag_err: exit(1)  # TODO


class TrueWRandomABSimulation(Simulation):
    def __init__(self, output_sim_dir, sub_dir_name='true_W_random_ab'):
        super().__init__(output_sim_dir, sub_dir_name=sub_dir_name)

    def setup(self, param_obj=None, data_obj=None):
        super().setup(param_obj=param_obj, data_obj=data_obj)
        self.start.W = self.true.W
        return self



########################################    Random   ########################################


class RandomWTrueABSimulation(Simulation):
    def __init__(self, output_sim_dir, sub_dir_name='random_W_true_ab'):
        super().__init__(output_sim_dir, sub_dir_name=sub_dir_name)

    def setup(self, param_obj=None, data_obj=None):
        super().setup(param_obj=param_obj, data_obj=data_obj)
        self.start.a = self.true.a
        self.start.b = self.true.b
        return self


class RandomParamsSimulation(Simulation):
    def __init__(self, output_sim_dir, sub_dir_name='EM'):
        super().__init__(output_sim_dir, sub_dir_name=sub_dir_name)


class RandomParamsSimulationRegW(Simulation):
    def __init__(self, output_sim_dir, sub_dir_name='EMRegW'):
        super().__init__(output_sim_dir, EM_type=EMRegW, sub_dir_name=sub_dir_name)



########################################    Known   ########################################


class KnownHSimulation(Simulation):
    def __init__(self, output_sim_dir, sub_dir_name='known_H'):
        super().__init__(output_sim_dir, EM_type=EMkH, sub_dir_name=sub_dir_name)

    def setup(self, param_obj=None, data_obj=None):
        super().setup(param_obj=param_obj, data_obj=data_obj)
        self.start.a = None
        self.start.b = None
        return self

    def _save_results(self):
        pd.DataFrame.to_csv(pd.DataFrame(self.ll_scores), self.sim_dir/'ll_scores.csv', index=False)
        self.prog.save_csv(self.sim_dir, prefix='hat_')
        self.prog.a = []
        self.prog.b = []


class KnownABSimulation(Simulation):
    def __init__(self, output_sim_dir, sub_dir_name='known_AB', EM_type=EMkAB):
        super().__init__(output_sim_dir, EM_type=EM_type, sub_dir_name=sub_dir_name)

    def _save_results(self):
        pd.DataFrame.to_csv(pd.DataFrame(self.ll_scores), self.sim_dir/'ll_scores.csv', index=False)
        self.prog.save_csv(self.sim_dir, prefix='hat_')
        self.prog.a = []
        self.prog.b = []


class KnownWSimulation(Simulation):
    def __init__(self, output_sim_dir, sub_dir_name='known_W', EM_type=EMkW):
        super().__init__(output_sim_dir, EM_type=EM_type, sub_dir_name=sub_dir_name)

    def setup(self, param_obj=None, data_obj=None):
        """
        hyper_params of param_obj: HyperParams(W:true_W, bg:bg=0), W = true W, will not change during the run
        """
        super().setup(param_obj=param_obj, data_obj=data_obj)
        self.hyper.W = data_obj.hyper.W
        return self

    def _save_results(self):
        pd.DataFrame.to_csv(pd.DataFrame(self.ll_scores), self.sim_dir/'ll_scores.csv', index=False)
        if self.prog.H is not None and len(self.prog.H) == 1: self.prog.H = []
        self.prog.save_csv(self.sim_dir, prefix='hat_')
        self.prog.W = []


########################################    NMF   ########################################



class NMFRandomParamsSimulation(Simulation):
    def __init__(self, output_sim_dir, sub_dir_name='NMF', EM_type=ExtendedNMFasEM):
        super().__init__(output_sim_dir, sub_dir_name=sub_dir_name, EM_type=EM_type)

    def _run_EM(self, em, eps, max_itr=0):
        if self.hat is None:
            W, H, ll_scores = em.fit()
            self.ll_scores = ll_scores
            self.hat = Params(W=W, H=H)
            self.prog = ParamsProg() + self.hat

        self._save_results()
        self._dump_simulation()

        return self.ll_scores, self.prog


class NMFKnownHSimulation(NMFRandomParamsSimulation):
    def __init__(self, output_sim_dir, sub_dir_name='NMFkH'):
        super().__init__(output_sim_dir, sub_dir_name=sub_dir_name, EM_type=ExtendedNMFasEMkH)



########################################    Batch   ########################################


class BatchRandomParamsSimulation(Simulation):
    def __init__(self, output_sim_dir, sub_dir_name='batchEM', EM_type=batchEM):
        super().__init__(output_sim_dir, sub_dir_name=sub_dir_name, EM_type=EM_type)
        self.itr_step_calc_ll = 1


class BatchKnownABSimulation(KnownABSimulation, BatchRandomParamsSimulation):
    def __init__(self, output_sim_dir, sub_dir_name='batchEMkAB', EM_type=batchEMkAB):
        super().__init__(output_sim_dir, sub_dir_name=sub_dir_name, EM_type=EM_type)


class BatchKnownWSimulation(KnownWSimulation, BatchRandomParamsSimulation):
    def __init__(self, output_sim_dir, sub_dir_name='batchEMkW', EM_type=batchEMkW):
        super().__init__(output_sim_dir, sub_dir_name=sub_dir_name, EM_type=EM_type)


########################################    Utils   ########################################



def run_simulation(simulator_type, sim_file_dir, data_obj=None, param_obj=None, sub_dir_name='', eps=.1, max_itr=0):
    simulator = simulator_type(sim_file_dir, sub_dir_name=sub_dir_name).setup(data_obj=data_obj, param_obj=param_obj)
    print('Running ', simulator, flush=True)
    simulator.run_simulation(eps=eps, max_itr=max_itr)
    print('*********', flush=True)
    return simulator

def load_simulation(sim_file_dir, simulator_type=Simulation, sub_dir_name=''):
    print('Loading simulation from', sim_file_dir)
    assert (sim_file_dir/'simulation_obj.pkl').exists(), f'Simulation file does not exists in {sim_file_dir}'
    simulator = simulator_type(sim_file_dir, sub_dir_name=sub_dir_name).setup()
    return simulator


def get_sim_params_by_itr(sim_dir, itr, sim=None):
    if sim is None: sim = load_simulation(sim_dir)
    return Params(W=sim.prog.W[itr] if len(sim.prog.W) > 0 else None,
                  a=sim.prog.a[itr] if len(sim.prog.a) > 0 else None,
                  b=sim.prog.b[itr] if len(sim.prog.b) > 0 else None)


def get_sim_params_by_thr(sim_dir, thr):
    sim = load_simulation(sim_dir)
    sim_ll = np.array(sim.ll_scores)
    idx = np.argwhere(sim_ll[1:] - sim_ll[:-1] < thr)[0][0]
    return get_sim_params_by_itr(sim_dir, idx, sim=sim)


def get_sim_hat_params(sim_dir):
    return get_sim_params_by_itr(sim_dir, -1)


