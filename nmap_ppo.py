import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import pdb
from nmap import NeuralMapPolicy
from PIL import Image

class NeuralMapModel(object):
    def __init__(self, *, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, nmap_args):
        sess = tf.get_default_session()

        # pass in architecture and also change NeuralMapPolicy to take these arguments
        act_model = NeuralMapPolicy(sess, ob_space, ac_space, nmap_args, nbatch_act, 1, False)
        train_model = NeuralMapPolicy(sess, ob_space, ac_space, nmap_args, nbatch_train, nsteps, True)
        A = train_model.pdtype.sample_placeholder([None], name='A')
        ADV = tf.placeholder(tf.float32, [None], name='ADV')
        R = tf.placeholder(tf.float32, [None], name='R')
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], name='OLDNEGLOGPAC')
        OLDVPRED = tf.placeholder(tf.float32, [None], name='OLDVPRED')
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = np.array(returns) - np.array(values)
            advs = (advs - advs.mean())/(advs.std() + 1e-8)
            obs_img, info = obs[0], obs[1]

            obs_img = np.array(list(obs_img))

            pos = [i['curr_loc'] for i in info]
            p_pos = [i['past_loc'] for i in info]
            step_counter = np.array([i['step_counter'] for i in info]).squeeze()

            td_map = {
                train_model.nmap.inputs: obs_img,
                train_model.nmap.pos: np.squeeze(pos, 1),
                train_model.nmap.p_pos: np.squeeze(p_pos, 1),
                train_model.nmap.timestep: np.expand_dims([t % nmap_args['max_timestep'] for t in step_counter], 1),
                train_model.nmap.masks: masks,
                CLIPRANGE: cliprange,
                LR: lr,
                OLDNEGLOGPAC: np.squeeze(neglogpacs),
                OLDVPRED: values,
                ADV: advs,
                R: returns,
                A: np.squeeze(actions)
            }
            if states is None:
                print('asdasdasd')
            else:
                memory, old_c_t, ctx_state = states

            td_map[train_model.nmap.memory] = memory
            td_map[train_model.nmap.old_c_t] = old_c_t
            td_map[train_model.nmap.ctx_state_tuple] = ctx_state

            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save_model_weights(path):
            # Helper function to save your model / weights.
            saver = tf.train.Saver()
            save_path = saver.save(sess, path)
            print("Model saved in path: %s"%(save_path))
        #def save(save_path):
            #ps = sess.run(params)
            #joblib.dump(ps, save_path)
        def load_model_weights(weights_file):
            # Helper funciton to load model weights.
            saver = tf.train.Saver()
            saver.restore(sess, weights_file)
            print("Model restored from %s"%weights_file)
        # def load(load_path):
        #     loaded_params = joblib.load(load_path)
        #     restores = []
        #     for p, loaded_p in zip(params, loaded_params):
        #         print (p)
        #         restores.append(p.assign(loaded_p))
        #     load_op=sess.run(restores)
        #     print ('Load output     ')
        #     print ('len',len(load_op))
        #     #for i in load_op:
        #         #print (load_op.shape)

            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.get_initial_state(nbatch_act)
        self.save = save_model_weights
        self.load = load_model_weights
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.nenv = nenv
        obs_img = env.reset()
        self.obs = [obs_img, env.initial_info() ]
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            self.obs, rewards, self.dones, infos = self.env.step(actions)
            if infos.get('episode'):
                epinfo_ = infos['episode']
                epinfo_ = [e for e in epinfo_ if e is not None]
                epinfos.extend(epinfo_)

            # mask and return model states
            self.states = self.model.act_model.get_initial_state(self.nenv,self.states,self.dones)
            self.obs = [self.obs, infos]

            mb_rewards.append(rewards)


        # split into tuple of (np.array(state observations), list of dictionaries)
        mb_obs = (np.asarray([m[0] for m in mb_obs]), [m[1] for m in mb_obs])
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32).squeeze()
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = np.asarray(self.model.act_model.value(self.obs, self.states, self.dones)).squeeze()

        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        return ((sf01(mb_obs[0]), sf01dict(mb_obs[1], self.nenv)), sf01(mb_returns), sf01(mb_dones), sf01(mb_actions), sf01(mb_values), sf01(mb_neglogpacs),
            mb_states, epinfos)

def sf01dict(arr, nenvs):
    flattened_list_dicts = []
    for env_ix in range(nenvs):
        for arr_item in arr:
            new_dict = {}
            for k in arr_item:
                if k == 'episode' and arr_item[k][env_ix] is not [None]: # TODO: janky!!!
                    new_dict['episode'] = arr_item['episode'][env_ix]
                else:
                    new_dict[k] = [arr_item[k][env_ix]]
            if 'episode' in new_dict and new_dict['episode'] is None:
                del new_dict['episode']
            flattened_list_dicts.append(new_dict)
    return flattened_list_dicts

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

def learn(*, env, nsteps, total_timesteps, ent_coef, lr, nmap_args,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
             nminibatches=4, noptepochs=1, cliprange=0.2,
            log_interval=1, save_interval=10,load=None):

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    make_model = lambda : NeuralMapModel(ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    nmap_args=nmap_args,
                    max_grad_norm=max_grad_norm)

    model = make_model()
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    if load==None:
        print("""Will save in %s""" % nmap_args['savepath'])
        #if save_interval and nmap_args['savepath']:
            #import cloudpickle
            #with open(osp.join(nmap_args['savepath'], 'make_model.pkl'), 'wb') as fh:
                #fh.write(cloudpickle.dumps(make_model))

        epinfobuf = deque(maxlen=100)
        tfirststart = time.time()

        nupdates = total_timesteps//nbatch
        for update in range(1, nupdates+1):
            print ('UPDATE NUMBER = ',update)
            assert nbatch % nminibatches == 0
            nbatch_train = nbatch // nminibatches
            tstart = time.time()
            frac = 1.0 - (update - 1.0) / nupdates
            lrnow = lr(frac)
            cliprangenow = cliprange(frac)
            obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
            epinfobuf.extend(epinfos)
            mblossvals = []
            if states is None: # nonrecurrent version
                inds = np.arange(nbatch)
                for _ in range(noptepochs):
                    np.random.shuffle(inds)
                    for start in range(0, nbatch, nbatch_train):
                        end = start + nbatch_train
                        mbinds = inds[start:end]
                        slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices))
            else: # recurrent version
                assert nenvs % nminibatches == 0
                mblossvals=[]
                cliprangenow=cliprange(frac)
                envsperbatch = nenvs//nminibatches
                envinds = np.arange(nenvs)
                flatinds = np.arange(nenvs*nsteps).reshape(nenvs,nsteps)
                envsperbatch = nbatch_train // nsteps

                states = states[:-1] # ignore the last state
                for _ in range(noptepochs):
                    np.random.shuffle(envinds)

                    for start in range(0, nenvs, envsperbatch):
                        end = start + envsperbatch
                        mbenvinds = envinds[start:end]
                        mbflatinds = flatinds[mbenvinds].ravel()
                        slices = list(arr[mbflatinds] for arr in (returns, masks, actions, values, neglogpacs))
                        slices.insert(0, (obs[0][mbflatinds], [obs[1][ix] for ix in mbflatinds]))

                        mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))




            lossvals = np.mean(mblossvals, axis=0)
            tnow = time.time()
            fps = int(nbatch / (tnow - tstart))
            if update % log_interval == 0 or update == 1:
                ev = explained_variance(values, returns)
                logger.logkv("serial_timesteps", update*nsteps)
                logger.logkv("nupdates", update)
                logger.logkv("total_timesteps", update*nbatch)
                logger.logkv("fps", fps)
                logger.logkv("explained_variance", float(ev))
                logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
                logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
                logger.logkv('time_elapsed', tnow - tfirststart)
                for (lossval, lossname) in zip(lossvals, model.loss_names):
                    logger.logkv(lossname, lossval)
                logger.dumpkvs()
            if save_interval and (update % save_interval == 0 or update == 1) and nmap_args['savepath']:
                checkdir = nmap_args['savepath']
                os.makedirs(checkdir, exist_ok=True)
                savepath = osp.join(checkdir, '%.5i'%update)
                print('Saving to', savepath)
                model.save(path=savepath)
    else:
        model.load(load)

        epinfobuf = deque(maxlen=100)
        rewards_list=[]
        nupdates = total_timesteps//nbatch
        for update in range(1, nupdates+1):
            obs_img = env.reset()
            obs = [obs_img, env.initial_info() ]
            states = model.initial_state
            dones=[False]

            print ('Test episode number = ',update)
            assert nbatch % nminibatches == 0
            nbatch_train = nbatch // nminibatches
            tstart = time.time()
            frac = 1.0 - (update - 1.0) / nupdates
            lrnow = lr(frac)
            cliprangenow = cliprange(frac)

            mb_rewards = []
            while not dones[-1]:
                actions, values, mem, old_c_t, ctx_state, neglogpacs = model.step(obs, states, dones)
                states = (mem, np.expand_dims(old_c_t, 1), ctx_state)
                obs, rewards, dones, infos = env.step(actions)
                env.envs[0].render()
                states = model.act_model.get_initial_state(1,states,dones)
                obs = [obs, infos]

                mb_rewards.append(rewards)

            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            rewards_list.append(np.sum(mb_rewards))
        for i in range(0,nupdates):
            print ('Test rewards for episode',i,'is= ',rewards_list[i])
        print ('Average test rewards = ',np.mean(rewards_list))

    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

class PacmanDummyVecEnv(DummyVecEnv):
    def step_wait(self):
        for i in range(self.num_envs):
            obs_tuple, self.buf_rews[i], self.buf_dones[i], self.buf_infos[i] = self.envs[i].step(self.actions[i])
            if self.buf_dones[i]:
                obs_tuple = self.envs[i].reset()
                buf_info = self.envs[i].initial_info
                buf_info['episode'] = self.buf_infos[i]['episode'] # explicit remembering epinfo (of prev episode)
                self.buf_infos[i] = buf_info.copy()
            if isinstance(obs_tuple, (tuple, list)):
                for t,x in enumerate(obs_tuple):
                    self.buf_obs[t][i] = x
            else:
                self.buf_obs[0][i] = obs_tuple

        self.info = self.buf_infos[0]
        for i in range(1, self.num_envs):
            for k in self.buf_infos[i]:
                if k in self.buf_infos[i] and k in self.info:
                    self.info[k].extend(self.buf_infos[i][k])
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.info.copy())

    def initial_info(self):
        initial_info_ = self.envs[0].initial_info
        for i in range(1, self.num_envs):
            for k in self.envs[i].initial_info:
                initial_info_[k].extend(self.envs[i].initial_info[k])
        return initial_info_


