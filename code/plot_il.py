from my_utils import *
from args_parser import * 
from core.agent import Agent

from core.dqn import *
from core.ac import *
from core.irl import *
from core.vild import *

import matplotlib.pyplot as plt
import pylab
#from https://tonysyu.github.io/plotting-error-bars.html#.WRwXWXmxjZs
def errorfill(x, y, yerr, color=None, alpha_fill=0.15, ax=None, linestyle="-", linewidth = None, label=None, shade=True):

    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, linestyle=linestyle, color=color, linewidth = linewidth, label=label)
    #ax.plot(x, y, pltcmd, linewidth = linewidth, label=label)
    if shade:
        ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

def running_mean_x(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

# Except the first element (initial policy)
def running_mean(x, N):
    x_tmp = x[1:-1]
    cumsum = np.cumsum(np.insert(x_tmp, 0, 0)) 
    tmp =  (cumsum[N:] - cumsum[:-N]) / N 
    return  np.insert(tmp, 0, x[0]) 

def load(filename, limit=1000, load_robo=0):
    if load_robo:
        try:
        # if 1:
            R_test_avg = np.load(filename + "_e100_test.npy") 
            return R_test_avg
        except:
            return np.reshape(np.array([-999]), (-1, 1))
    else:                
        R_test_avg = []
        step_idx = -1
        prev_step = 0
        try:
            with open(filename + ".txt", 'r') as f:
                for line in f:
                    line = line.replace(":", " ").replace("(", " ").replace(")", " ").replace(",", " ").split()
                    if step_idx == -1:
                        step_idx = line.index("Step") + 1
                        R_test_avg_idx = line.index("[R_te]") + 6
                    cur_step = int(line[step_idx]) 
                    if cur_step < prev_step:
                        print("reset array at %s" % filename)
                        R_test_avg = [] 
                    R_test_avg += [ float(line[R_test_avg_idx]) ]
                    prev_step = cur_step 
            R_test_avg = np.reshape(np.array(R_test_avg), (-1, 1))   # [iter , 1] array
            return R_test_avg
        except  Exception as e:
            print(e)
            return np.reshape(np.array([-999]), (-1, 1))

def load_noise(filename):
    fileHandle = open ( filename + ".txt" ,"r" )
    lineList = fileHandle.readlines()
    fileHandle.close()
    
    lastline =  lineList[len(lineList)-1]
    lastline = lastline.replace(":", " ").replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ").split()

    noise_line_idx = lastline.index("w_noise") + 1

    noise_list = []
    for k in range(0, 10):
        noise_list += [ float(lastline[noise_line_idx + k]) ]

    # tmp = np.reshape(np.array(noise_list), (-1, 1)) 

    return np.reshape(np.array(noise_list), (-1, 1)) 

def load_info(filename, limit=1000, load_robo=0):
    if load_robo:
        # try:
        if 1:
            R_test_avg = [[] for i in range(0, 10)]
            for i_c in range(0, 10):
                R_test_avg[i_c] = np.load(filename + "_e10_c%d_test.npy" % i_c) 
            
            R_test_avg = np.array(R_test_avg)   #  [code_dim(10), test_iter, test_epi]            
            R_test_avg = np.transpose(R_test_avg, (1,0,2)) # [test_iter, code_dim, test_epi] 

            return R_test_avg

        # except:
        #     return np.reshape(np.array([-999]), (-1, 1))
    else:                
        R_test_avg = [[] for i in range(0, 10)]
        step_idx = -1
        prev_step = 0
        try:
            with open(filename + ".txt", 'r') as f:
                for line in f:
                    line = line.replace(":", " ").replace("(", " ").replace(")", " ").replace(",", " ").split()
                    if step_idx == -1:
                        R_idx = [] 
                        step_idx = line.index("Step") + 1
                        ii = line.index("Avg")
                        for i_w in range(0, 10):  # 10 workers. 
                            ii = ii + 2
                            R_idx += [ii]
                            ii = ii + 1

                    cur_step = int(line[step_idx]) 
                    if cur_step < prev_step:
                        print("reset array at %s" % filename)
                        R_test_avg = [[] for i in range(0, 10)]

                    for i_w in range(0, 10):
                        R_test_avg[i_w] += [ float(line[R_idx[i_w]]) ]

                    prev_step = cur_step 

            R_test_avg = np.array(R_test_avg)   #  [code_dim (10), test_iter]
            R_test_avg = np.transpose(R_test_avg, (1,0))  # [test_iter , code_dim (10) ] 
            R_test_avg = np.expand_dims(R_test_avg, axis = -1)  # [test_iter , code_dim (10), test_epi_recorded (1) ] 
            return R_test_avg
        except  Exception as e:
            print(e)
            return np.reshape(np.array([-999]), (-1, 1))

def plot(args):

    limit = 1000
    env_name = args.env_name 
    if args.env_id == 9:
        limit = 500 

    """ get config about trajectory file"""
    
    if args.env_id != 21:
        discriminator_updater = IRL(1, 1, args, False)
        traj_name = discriminator_updater.traj_name
        m_return_list = discriminator_updater.m_return_list
    else:
        traj_name = "traj10"
        m_return_list = None 

    seed_list = [1, 2, 3, 4, 5]
    seed_num = len(seed_list)
    print(seed_list)
    load_robo = 0 

    plot_methods = ["vild_per0", "vild_per2", "meirl", "airl", "gail", "vail", "infogail_mean", "infogail_best"]
    if args.env_id == 9:
        args.info_loss_type = "linear"
        args.rl_method = "sac"
        args.hidden_size = (100, 100)
        args.activation = "relu"
        args.q_step = 1

    if args.env_id == 0:
        plot_methods = ["vild_per0", "vild_per2", "gail", "infogail_mean", "infogail_best"]
        args.vild_loss_type = "bce"

    if args.env_id == -3:
        plot_methods = ["vild_per0", "vild_per2", "gail", "infogail_mean", "infogail_best"]
        args.vild_loss_type = "bce"
        args.rl_method = "ppo"
        args.bce_negative = 1
        args.hidden_size = (100, 100)
        args.activation = "relu"
        args.gp_lambda = 0

    if args.env_id == 21:
        plot_methods = ["vild_per0", "vild_per2", "gail", "infogail_mean", "infogail_best", "meirl", "airl", "vail"]    # for paper 
        # plot_methods = ["vild_per2", "gail", "meirl", "airl", "vail"]   # for slide
        args.vild_loss_type = "bce"
        env_name += "_reach" 
        args.hidden_size = (100, 100)
        args.activation = "relu"
        load_robo = 1

    if args.env_id == 15:
        plot_methods = ["vild_per2", "gail", "meirl"]
        # args.hidden_size = (100, 100)
        # args.activation = "tanh"

    R_test_all = []
    gail_legend = []
    c_tmp = []
    l_tmp = []

    max_len = 10000

    if any("vild_per2" in s for s in plot_methods):
        """ VILD """
        R_test_avg = []
        cat = 1
        noise_array_is = []
        args.il_method = "vild"
        args.per_alpha = 2
        args.clip_discriminator = 5   
        method_name = args.il_method.upper() + "_" + args.rl_method.upper() 
        if args.vild_loss_type.lower() != "linear": 
            method_name += "_" + args.vild_loss_type.upper()
            args.clip_discriminator = 0
        hypers = rl_hypers_parser(args) + "_" + irl_hypers_parser(args)      
        for seed in seed_list:
            exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers, seed)
            if load_robo:
                filename = "./results_IL/test/%s/%s-%s" % (env_name, env_name, exp_name)
            else:
                filename = "./results_IL/%s/%s/%s-%s" % (method_name, env_name, env_name, exp_name)

            R_test_avg_i = load(filename, limit, load_robo)
            if not load_robo and args.c_data == 7:
                noise_array_is += [load_noise(filename)]
            if R_test_avg_i[0, 0] == -999:
                cat = 0
                print("cannot load %s" % exp_name) 
            else:
                load_iter = R_test_avg_i.shape[0]
                if load_iter < max_len:
                    max_len = load_iter 
                    for i in range(0, seed-1):
                        R_test_avg[i] = R_test_avg[i][0:max_len, :]
                R_test_avg_i = R_test_avg_i[0:max_len, :]
                R_test_avg += [R_test_avg_i]
        if cat:
            R_test_avg = np.concatenate(R_test_avg, axis=1) # array [iter, test_episode * seed]
            R_test_all += [R_test_avg]
            gail_legend += ["VILD (IS)"]
            c_tmp += ["r"]
            l_tmp += ["-"]

    if any("vild_per0" in s for s in plot_methods):
        """ VILD no IS """
        R_test_avg = []
        cat = 1
        noise_array = []
        args.il_method = "vild"
        args.per_alpha = 0
        args.clip_discriminator = 5 
        method_name = args.il_method.upper() + "_" + args.rl_method.upper() 
        if args.vild_loss_type.lower() != "linear": 
            method_name += "_" + args.vild_loss_type.upper()  
            args.clip_discriminator = 0
        hypers = rl_hypers_parser(args) + "_" + irl_hypers_parser(args)       
        for seed in seed_list:
            exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers, seed)
            if load_robo:
                filename = "./results_IL/test/%s/%s-%s" % (env_name, env_name, exp_name)
            else:
                filename = "./results_IL/%s/%s/%s-%s" % (method_name, env_name, env_name, exp_name)

            R_test_avg_i = load(filename, limit, load_robo)
            if not load_robo:
                noise_array += [load_noise(filename)]
            if R_test_avg_i[0, 0] == -999:
                cat = 0
                print("cannot load %s" % exp_name) 
            else:
                load_iter = R_test_avg_i.shape[0]
                if load_iter < max_len:
                    max_len = load_iter 
                    for i in range(0, seed-1):
                        R_test_avg[i] = R_test_avg[i][0:max_len, :]
                R_test_avg_i = R_test_avg_i[0:max_len, :]
                R_test_avg += [R_test_avg_i]
        if cat:
            R_test_avg = np.concatenate(R_test_avg, axis=1) # array [iter, test_episode * seed]
            R_test_all += [R_test_avg]
            gail_legend += ["VILD (w/o IS)"]  
                    
            c_tmp += ["blueviolet"]
            l_tmp += ["-"]

    if any("gail" in s for s in plot_methods):
        """ GAIL """
        R_test_avg = []
        cat = 1
        args.il_method = "gail"
        args.clip_discriminator = 0 
        hypers = rl_hypers_parser(args) + "_" + irl_hypers_parser(args)        
        method_name = args.il_method.upper() + "_" + args.rl_method.upper() 
        for seed in seed_list:
            exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers, seed)
            if load_robo:
                filename = "./results_IL/test/%s/%s-%s" % (env_name, env_name, exp_name)
            else:
                filename = "./results_IL/%s/%s/%s-%s" % (method_name, env_name, env_name, exp_name)

            R_test_avg_i = load(filename, limit, load_robo)
            if R_test_avg_i[0, 0] == -999:
                cat = 0
                print("cannot load %s" % exp_name) 
            else:
                load_iter = R_test_avg_i.shape[0]
                if load_iter < max_len:
                    max_len = load_iter 
                    for i in range(0, seed-1):
                        R_test_avg[i] = R_test_avg[i][0:max_len, :]
                R_test_avg_i = R_test_avg_i[0:max_len, :]
                R_test_avg += [R_test_avg_i]
        if cat:
            R_test_avg = np.concatenate(R_test_avg, axis=1) # array [iter, test_episode * seed]
            R_test_all += [R_test_avg]
            gail_legend += ["GAIL"]
            c_tmp += ["darkgoldenrod"]
            l_tmp += ["--"]

    if any("airl" in s for s in plot_methods):
        """ AIRL """
        R_test_avg = []
        cat = 1
        args.il_method = "airl"
        args.clip_discriminator = 5 
        hypers = rl_hypers_parser(args) + "_" + irl_hypers_parser(args)        
        method_name = args.il_method.upper() + "_" + args.rl_method.upper() 
        for seed in seed_list:
            exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers, seed)
            if load_robo:
                filename = "./results_IL/test/%s/%s-%s" % (env_name, env_name, exp_name)
            else:
                filename = "./results_IL/%s/%s/%s-%s" % (method_name, env_name, env_name, exp_name)

            R_test_avg_i = load(filename, limit, load_robo)
            if R_test_avg_i[0, 0] == -999:
                cat = 0
                print("cannot load %s" % exp_name) 
            else:
                load_iter = R_test_avg_i.shape[0]
                if load_iter < max_len:
                    max_len = load_iter 
                    for i in range(0, seed-1):
                        R_test_avg[i] = R_test_avg[i][0:max_len, :]
                R_test_avg_i = R_test_avg_i[0:max_len, :]
                R_test_avg += [R_test_avg_i]
        if cat:
            R_test_avg = np.concatenate(R_test_avg, axis=1) # array [iter, test_episode * seed]
            R_test_all += [R_test_avg]
            gail_legend += ["AIRL"]
            c_tmp += ["b"]
            l_tmp += ["-."]

    if any("meirl" in s for s in plot_methods):
        """ ME-IRL """
        R_test_avg = []
        cat = 1
        args.il_method = "irl"
        args.clip_discriminator = 5 
        hypers = rl_hypers_parser(args) + "_" + irl_hypers_parser(args)        
        method_name = args.il_method.upper() + "_" + args.rl_method.upper() 
        for seed in seed_list:
            exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers, seed)
            if load_robo:
                filename = "./results_IL/test/%s/%s-%s" % (env_name, env_name, exp_name)
            else:
                filename = "./results_IL/%s/%s/%s-%s" % (method_name, env_name, env_name, exp_name)

            R_test_avg_i = load(filename, limit, load_robo)
            if R_test_avg_i[0, 0] == -999:
                cat = 0
                print("cannot load %s" % exp_name) 
            else:
                load_iter = R_test_avg_i.shape[0]
                if load_iter < max_len:
                    max_len = load_iter 
                    for i in range(0, seed-1):
                        R_test_avg[i] = R_test_avg[i][0:max_len, :]
                R_test_avg_i = R_test_avg_i[0:max_len, :]
                R_test_avg += [R_test_avg_i]
        if cat:
            R_test_avg = np.concatenate(R_test_avg, axis=1) # array [iter, test_episode * seed]
            R_test_all += [R_test_avg]
            gail_legend += ["ME-IRL"]
            c_tmp += ["olive"]
            l_tmp += [":"]

    if any("vail" in s for s in plot_methods):
        """ VAIL """
        R_test_avg = []
        cat = 1
        args.il_method = "vail"
        args.clip_discriminator = 0 
        hypers = rl_hypers_parser(args) + "_" + irl_hypers_parser(args)        
        method_name = args.il_method.upper() + "_" + args.rl_method.upper() 
        for seed in seed_list:
            exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers, seed)
            if load_robo:
                filename = "./results_IL/test/%s/%s-%s" % (env_name, env_name, exp_name)
            else:
                filename = "./results_IL/%s/%s/%s-%s" % (method_name, env_name, env_name, exp_name)

            R_test_avg_i = load(filename, limit, load_robo)
            if R_test_avg_i[0, 0] == -999:
                cat = 0
                print("cannot load %s" % exp_name) 
            else:
                load_iter = R_test_avg_i.shape[0]
                if load_iter < max_len:
                    max_len = load_iter 
                    for i in range(0, seed-1):
                        R_test_avg[i] = R_test_avg[i][0:max_len, :]
                R_test_avg_i = R_test_avg_i[0:max_len, :]
                R_test_avg += [R_test_avg_i]
        if cat:
            R_test_avg = np.concatenate(R_test_avg, axis=1) # array [iter, test_episode * seed]
            R_test_all += [R_test_avg]
            gail_legend += ["VAIL"]
            c_tmp += ["m"]
            l_tmp += ["--"]

    if any("infogail_mean" in s for s in plot_methods):
        """ InfoGAIL """
        R_test_avg = []
        cat = 1
        args.il_method = "infogail"
        args.clip_discriminator = 5 if args.info_loss_type == "linear" else 0 
        hypers = rl_hypers_parser(args) + "_" + irl_hypers_parser(args)        
        method_name = args.il_method.upper() + "_" + args.rl_method.upper() 
        if args.il_method == "infogail" and args.info_loss_type.lower() != "bce":
            method_name += "_" + args.info_loss_type.upper()

        for seed in seed_list:
            exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers, seed)
            if load_robo:
                filename = "./results_IL/test/%s/%s-%s" % (env_name, env_name, exp_name)
            else:
                filename = "./results_IL/%s/%s/%s-%s" % (method_name, env_name, env_name, exp_name)

            R_test_avg_i = load_info(filename, limit, load_robo)  # [iter, 10, test_episode (1?) ] array 
            if R_test_avg_i[0, 0, 0] == -999:
                cat = 0
                print("cannot load %s" % exp_name) 
            else:
                load_iter = R_test_avg_i.shape[0]
                if load_iter < max_len:
                    max_len = load_iter 
                    for i in range(0, seed-1):
                        R_test_avg[i] = R_test_avg[i][0:max_len, :]
                R_test_avg_i = R_test_avg_i[0:max_len, :]
                R_test_avg += [R_test_avg_i]
        if cat:
            # R_test_avg_x = np.stack(R_test_avg, axis=2) # array [iter, 10, test_episode * seed]
            R_test_avg_x = np.concatenate(R_test_avg, axis=2 )

            # average over uniform contexts
            R_test_avg_mean = np.mean(R_test_avg_x, axis=1)       
            R_test_all += [R_test_avg_mean]
            gail_legend += ["InfoGAIL"]
            c_tmp += ["black"]
            l_tmp += ["-."]

        if any("infogail_best" in s for s in plot_methods):
            R_test_avg_sort = []
            ## For each seed, sort array along context descending based on performance
            for i in range(0, len(seed_list)):
                R_test_x = R_test_avg[i] # R_test_avg is a list. R_test_x is a np array shape [test_iter, 10, test_epi ]
                test_last = np.mean(np.mean(R_test_x, axis=2)[-100:,:], axis=0)  #  final performance average over last 100 iterations 
                sort_idx = np.argsort(test_last)[::-1]  # descending indices
                R_test_x_sort = R_test_x[:, sort_idx, :]
                R_test_avg_sort += [R_test_x_sort]

            R_test_avg_sort = np.concatenate(R_test_avg_sort, axis=2) # array [iter, 10, test_episode * seed]

            R_test_avg_sort_best = R_test_avg_sort[:, 0, :]      
            R_test_all += [R_test_avg_sort_best]
            gail_legend += ["InfoGAIL (best)"]
            c_tmp += ["dodgerblue"]
            l_tmp += [":"]

        if any("infogail_worst" in s for s in plot_methods):

            R_test_avg_sort_worst = R_test_avg_sort[:, -1, :]      
            R_test_all += [R_test_avg_sort_worst]
            gail_legend += ["InfoGAIL (worst)"]
            c_tmp += ["teal"]
            l_tmp += ["--"]

        ## infogail_all 
        R_test_info = []
        R_test_info_mean = []
        R_test_info_std = []
        for i in range(0, 10):
            R_test_info += [R_test_avg_sort[:, i, :]]
            R_test_info_mean += [np.mean(R_test_info[i], 1)]
            R_test_info_std += [np.std(R_test_info[i], 1)/np.sqrt(R_test_info[i].shape[1])]

    """ Compute statistics for plotting """
    R_test_mean = []
    R_test_std = []
    R_test_last = []
    R_test_last_mean = []
    R_test_last_std = []
    best_idx = -1 
    best_R = -1e6
    for i in range(0, len(R_test_all)):
        R_test = R_test_all[i]

        """ This is for plotting """
        R_test_mean += [np.mean(R_test, 1)]
        R_test_std += [np.std(R_test, 1)/np.sqrt(R_test.shape[1])]
    
        ## Average last final 100 iteration. 
        R_test_last += [np.mean(R_test[-100:,:], 0)]

        last_mean = np.mean(R_test_last[i], 0)
        if last_mean > best_R:
            best_idx = i 
            best_R = last_mean 

        R_test_last_mean += [last_mean]
        R_test_last_std += [np.std(R_test_last[i], 0)/np.sqrt(R_test.shape[1])]

        ### Print statistics 
        # print("%s: %0.1f(%0.1f)" % (gail_legend[i], R_test_mean_last, R_test_std_last))

    """ For t_test """
    ## paired t-test
    from scipy import stats
    
    best_m = R_test_last[best_idx]
    p_list = []
    for i in range(0, len(R_test_all)):
        # if gail_legend[i] == "InfoGAIL (best)": continue 
        if i != best_idx:
            _, p = stats.ttest_ind(best_m, R_test_last[i], nan_policy="omit")
        else:
            p = 1        
        p_list += [p]
        
    ## copied able latex format
    latex = ""
    for i in range(0, len(R_test_all)):
        # if gail_legend[i] == "InfoGAIL (best)": continue 
        print("%-70s:   Sum %0.0f(%0.0f) with p-value %f" % (gail_legend[i], R_test_last_mean[i], R_test_last_std[i], p_list[i]))
        if p_list[i] > 0.01:    # 1 percent confidence 
            latex += " & \\textbf{%0.0f (%0.0f)}" % (R_test_last_mean[i], R_test_last_std[i])
        else:
            latex += " & %0.0f (%0.0f)" % (R_test_last_mean[i], R_test_last_std[i])
    print(latex + " \\\\")

    plot_large = args.plot_large
    skipper = 10  # Use sliding window to compute running mean and std for clearer figures. skipper 1 = no running.
    running = 1
    plot_name = "%s_%s_%s" % (args.rl_method, env_name, args.noise_type )
    title = "%s" % (env_name[:-3])
    x_max_iter = len(R_test_mean[0])

    """ Plot """
    if plot_large: 
        linewidth = 2
        fontsize = 21
        f = plt.figure(figsize=(8, 6)) # for Figure 2, ...
    else:
        linewidth = 1 # for Figure 2, ...
        fontsize = 14   # for Figure 2, ...
        f = plt.figure(figsize=(4, 3)) # for Figure 2, ...

    ax = f.gca()

    """ Plot """
    cc_tmp =  ["black"] + ["gray"] + ["darkgray"] + ["dimgray"] + ["silver"]
     
    for i in range(0, len(R_test_all)):
        if running:
            y_plot = running_mean(R_test_mean[i][:x_max_iter], skipper)
            y_err = running_mean(R_test_std[i][:x_max_iter], skipper)
        else:
            y_plot = R_test_mean[i][:x_max_iter:skipper]
            y_err = R_test_std[i][:x_max_iter:skipper]

        x_ax = np.arange(0, len(y_plot)) * 10000 // 2
        if args.rl_method == "sac":
            x_ax = np.arange(0, len(y_plot)) * 10000 // 2 // 5
        if args.env_id == 21:
            x_ax = np.arange(0, len(y_plot)) * 100000 * 5 / 4
            
        if i == 0 and m_return_list is not None :
            ic_k = 0
            for i_k in range(0, len(m_return_list)):
                opt_plot = (x_ax * 0) + m_return_list[i_k]
                if i_k == 0 or i_k == 2 or i_k == 4 or i_k == 6 or i_k == 9:
                # if i_k == 0 or i_k == 4 or i_k == 8:
                    ax.plot(x_ax, opt_plot, color=cc_tmp[ic_k], linestyle=":", linewidth=linewidth+0.5)
                    # plt.text(x=0, y=m_return_list[i_k], s="k=%d" % (i_k+1), fontsize=fontsize-3, color=cc_tmp[ic_k])
                    ic_k += 1
        errorfill(x_ax, y_plot, yerr=y_err, color=c_tmp[i], linestyle=l_tmp[i], linewidth=linewidth, label=gail_legend[i], shade=1)

    if plot_large == 1 :
        if args.env_id == 0:
            ax.legend(prop={"size":fontsize-3}, frameon=True, loc = 'lower right', framealpha=1, ncol=2)        
        elif args.env_id == -3:
            # ax.legend(prop={"size":fontsize-4}, frameon=True, loc = 'lower right', framealpha=1, ncol=2)      
            ax.legend(prop={"size":fontsize-3}, frameon=True, loc = 'lower right', framealpha=1, ncol=2)      
        elif args.env_id == 21:
            ax.legend(prop={"size":fontsize-6.3}, frameon=True, loc = 'lower right', framealpha=1, ncol=3)     
            # handles, labels = ax.get_legend_handles_labels()
            # leg_order = [0, 1, 2, 3, 4, 5, 6]
            # leg_order = [0, 1, 6, 3, 4, 5, 2]
            # handles, labels = [handles[i] for i in leg_order], [labels[i] for i in leg_order]
            # ax.legend(handles, labels, prop={"size":fontsize-4}, frameon=True, loc = 'lower right', framealpha=1, ncol=3)
        else:
            ax.legend(prop={"size":fontsize-3}, frameon=True, loc = 'upper left', framealpha=1)
    else:
        if args.env_id == -3:
            ax.legend(prop={"size":fontsize-3}, frameon=True, loc = 'lower right', framealpha=1, ncol=2)      
        if args.env_id == 21:
            ax.legend(prop={"size":fontsize-6}, frameon=True, loc = 'lower right', framealpha=1, ncol=3)     


    if args.env_id == -3:
        title = "%s" % ("LunarLander")
        # ax.set_ylim([-200,250])
        # ax.set_ylim([-200,300])
        # f.legend(prop={"size":fontsize-3}, frameon=True, loc="lower right", framealpha=1, ncol=3)

    if args.env_id == 21:
        title = "%s" % ("RobosuiteReacher")
        plot_name = "%s_%s" % (args.rl_method, env_name )
        ax.set_ylim([-10, 45])  # paper
        # ax.set_ylim([-2.4, 29])    # slide 
            
    if args.env_id == 15:
        title = "%s" % ("AntBullet")        
                                           
    fig_name = "./figures/%s" % (plot_name)
    if plot_large: 
        fig_name += "_large"
    
    fig_name += "_slide"
    print(fig_name) 

    if plot_large:
        plt.title(title, fontsize=fontsize+1)
    else:
        plt.title(title, fontsize=fontsize)
            
    plt.xlabel("Transition samples", fontsize=fontsize-2)      
    plt.ylabel('Cumulative rewards', fontsize=fontsize-2)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    ## plot legend.
    if plot_large == 2:
        figlegend = pylab.figure(figsize=(18,1))
        #pylab.figlegend(*ax.get_legend_handles_labels(), prop={"size":fontsize-3}, ncol=7, handlelength=2.8, )
        pylab.figlegend(*ax.get_legend_handles_labels(), prop={"size":fontsize-3}, ncol=10, handlelength=2.0, )       
        figlegend.savefig("./figures/legend.pdf", bbox_inches='tight', pad_inches=-0.25)

    if args.plot_save:
        f.savefig(fig_name + ".pdf", bbox_inches='tight', pad_inches = 0)
    
    ## plot covariance.
    if not load_robo and args.c_data == 7:
        f_cov = plt.figure(figsize=(4, 3)) # for Figure 2, ...
        ax_cov = plt.gca()

        noise_mean_is = np.mean(np.array(noise_array_is), axis=0)
        noise_mean = np.mean(np.array(noise_array), axis=0)

        true_variance = np.array(args.noise_level_list) ** 2 # convert to variance

        if args.env_id == -3:
            1 
            # true_variance = [0.021, 0.022, 0.023, 0.024, 0.025,  0.026, 0.027, 0.028, 0.029, 0.030]
            # true_variance = true_variance * 0.034
        
        x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        
        x_pos = [i for i, _ in enumerate(x)]

        plt.ylabel('Covariance', fontsize=fontsize-2)
        plt.xlabel('Demonstrator number k', fontsize=fontsize-2)

        plt.plot(x_pos, noise_mean_is, 'rx', markersize=8, label="VILD (IS)")
        plt.plot(x_pos, noise_mean, 'm+', markersize=8, label="VILD (w/o IS)")
        
        if args.env_id == -3 :
            plt.legend(prop={"size":fontsize-3}, frameon=True, loc='lower right', framealpha=1)
        elif args.noise_type == "SDNTN":
            plt.legend(prop={"size":fontsize-3}, frameon=True, loc='upper left', framealpha=1)
        else:
            plt.plot(x_pos, true_variance, 'ko', markersize=5, label="Ground-truth")
            plt.legend(prop={"size":fontsize-3}, frameon=True, loc='upper left', framealpha=1)
            

        plt.yticks(fontsize=fontsize-2)
        plt.tight_layout()
        plt.xticks(x_pos, x, fontsize=fontsize-2)

        fig_name_cov = "./figures/%s_cov" % (plot_name)

        if plot_large:
            plt.title(title + " (Covariance)", fontsize=fontsize+1)
        else:
            plt.title(title + " (Covariance)", fontsize=fontsize)
                
        print(fig_name_cov) 

        if args.plot_save:
            f_cov.savefig(fig_name_cov + ".pdf", bbox_inches='tight', pad_inches = 0)

    ## plot bc. 
    if args.env_id == 2 or args.env_id == 5 or args.env_id == 7 or args.env_id == 9:    # mujoco tasks. 
        c_tmp_2 = []
        l_tmp_2 = []
        R_test_all_2 = []
        gail_legend_2 = []

        """ BC """
        R_test_avg = []
        cat = 1
        method_name = "BC"
        for seed in seed_list:
            hypers_tmp = bc_hypers_parser(args) 
            exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers_tmp, seed)
            filename = "./results_IL/%s/%s/%s-%s" % (method_name, env_name, env_name, exp_name)
            R_test_avg_i = load(filename, limit, load_robo)
            if R_test_avg_i[0, 0] == -999:
                cat = 0
                print("cannot load %s" % exp_name) 
            else:
                load_iter = R_test_avg_i.shape[0]
                if load_iter < max_len:
                    max_len = load_iter 
                    for i in range(0, seed-1):
                        R_test_avg[i] = R_test_avg[i][0:max_len, :]
                R_test_avg_i = R_test_avg_i[0:max_len, :]
                R_test_avg += [R_test_avg_i]
        if cat:
            R_test_avg = np.concatenate(R_test_avg, axis=1) # array [iter, test_episode * seed]
            R_test_all_2 += [R_test_avg]
            gail_legend_2 += ["BC"]  
                    
            c_tmp_2 += ["blue"]
            l_tmp_2 += ["--"]

        """ D-BC """
        R_test_avg = []
        cat = 1
        method_name = "DBC"
        for seed in seed_list:
            hypers_tmp = bc_hypers_parser(args) 
            exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers_tmp, seed)
            filename = "./results_IL/%s/%s/%s-%s" % (method_name, env_name, env_name, exp_name)
            R_test_avg_i = load(filename, limit, load_robo)
            if R_test_avg_i[0, 0] == -999:
                cat = 0
                print("cannot load %s" % exp_name) 
            else:
                load_iter = R_test_avg_i.shape[0]
                if load_iter < max_len:
                    max_len = load_iter 
                    for i in range(0, seed-1):
                        R_test_avg[i] = R_test_avg[i][0:max_len, :]
                R_test_avg_i = R_test_avg_i[0:max_len, :]
                R_test_avg += [R_test_avg_i]
        if cat:
            R_test_avg = np.concatenate(R_test_avg, axis=1) # array [iter, test_episode * seed]
            R_test_all_2 += [R_test_avg]
            gail_legend_2 += ["BC-D"]  
                    
            c_tmp_2 += ["orange"]
            l_tmp_2 += ["-."]

        """ Co-BC """
        R_test_avg = []
        cat = 1
        method_name = "COBC"
        for seed in seed_list:
            hypers_tmp = bc_hypers_parser(args) 
            exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers_tmp, seed)
            filename = "./results_IL/%s/%s/%s-%s" % (method_name, env_name, env_name, exp_name)
            R_test_avg_i = load(filename, limit, load_robo)
            if R_test_avg_i[0, 0] == -999:
                cat = 0
                print("cannot load %s" % exp_name) 
            else:
                load_iter = R_test_avg_i.shape[0]
                if load_iter < max_len:
                    max_len = load_iter 
                    for i in range(0, seed-1):
                        R_test_avg[i] = R_test_avg[i][0:max_len, :]
                R_test_avg_i = R_test_avg_i[0:max_len, :]
                R_test_avg += [R_test_avg_i]
        if cat:
            R_test_avg = np.concatenate(R_test_avg, axis=1) # array [iter, test_episode * seed]
            R_test_all_2 += [R_test_avg]
            gail_legend_2 += ["Co-teaching"]  
                    
            c_tmp_2 += ["darkgreen"]
            l_tmp_2 += [":"]

        R_test_mean_2 = []
        R_test_std_2 = []
        for i in range(0, len(R_test_all_2)):
            R_test = R_test_all_2[i]

            """ This is for plotting """
            R_test_mean_2 += [np.mean(R_test, 1)]
            R_test_std_2 += [np.std(R_test, 1)/np.sqrt(R_test.shape[1])]
            
        if plot_large: 
            linewidth = 2
            fontsize = 21
            f_2 = plt.figure(figsize=(8, 6)) # for Figure 2, ...
        else:
            linewidth = 1 # for Figure 2, ...
            fontsize = 14   # for Figure 2, ...
            f_2 = plt.figure(figsize=(4, 3)) # for Figure 2, ...

        ax_2 = f_2.gca()
        from matplotlib import cm

        for i in range(0, len(R_test_all_2)):
            if running:
                y_plot = running_mean(R_test_mean_2[i][:x_max_iter], skipper)
                y_err = running_mean(R_test_std_2[i][:x_max_iter], skipper)
            else:
                y_plot = R_test_mean_2[i][:x_max_iter:skipper]
                y_err = R_test_std_2[i][:x_max_iter:skipper]

            x_ax = np.arange(0, len(y_plot)) * 10000 // 2 // 5

            if i== 0:
                errorfill(x_ax, (x_ax * 0) + R_test_last_mean[0], yerr=(x_ax * 0) + R_test_last_std[0], color=c_tmp[0], linestyle="-", linewidth=linewidth, shade=1)
 

            errorfill(x_ax, y_plot, yerr=y_err, color=c_tmp_2[i], linestyle=l_tmp_2[i], linewidth=linewidth, label=gail_legend_2[i], shade=1)

        # if plot_large:
        #     ax_2.legend(prop={"size":fontsize-3}, frameon=True, loc = 'lower right', framealpha=1, ncol=2)
        
        if args.env_id == 2:
            ax_2.legend(prop={"size":fontsize-3}, frameon=True, loc = 'lower right', framealpha=1, ncol=2)      

        if args.env_id == 2:
            ax_2.set_ylim([-999, 5000])

        fig_name_bc = "./figures/%s_bc" % (plot_name)
        if plot_large: 
            fig_name_bc += "_large"
        print(fig_name_bc) 

        # title += " (Supervised)"
        if plot_large:
            plt.title(title, fontsize=fontsize+1)
        else:
            plt.title(title, fontsize=fontsize)
                
        plt.xlabel("Gradient steps", fontsize=fontsize-2)      
        plt.ylabel('Cumulative rewards', fontsize=fontsize-2)
        plt.xticks(fontsize=fontsize-2)
        plt.yticks(fontsize=fontsize-2)
        plt.tight_layout()
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        ## plot legend.
        if plot_large == 2:
            figlegend = pylab.figure(figsize=(18,1))
            pylab.figlegend(*ax_2.get_legend_handles_labels(), prop={"size":fontsize-3}, ncol=10, handlelength=2.0, )       
            figlegend.savefig("./figures/legend_bc.pdf", bbox_inches='tight', pad_inches=-0.25)

        if args.plot_save:
            f_2.savefig(fig_name_bc + ".pdf", bbox_inches='tight', pad_inches = 0)

    ## plot infogail sorted performance. 
    if any("infogail_mean" in s for s in plot_methods):
        if plot_large: 
            linewidth = 2
            fontsize = 21
            f_3 = plt.figure(figsize=(8, 6)) # for Figure 2, ...
        else:
            linewidth = 1 # for Figure 2, ...
            fontsize = 14   # for Figure 2, ...
            f_3 = plt.figure(figsize=(4, 3)) # for Figure 2, ...
        ax_3 = f_3.gca()
        # info_color = ["black"] + ["gray"] + ["darkgray"] + ["dimgray"] + ["silver"] \
                    # + ["r"] + ["g"] + ["b"] + ["m"] + ["c"]
        from matplotlib import cm
        info_color = cm.gnuplot2(np.linspace(0,1,12))

        for i in range(0, 10):
            if running:
                y_plot = running_mean(R_test_info_mean[i][:x_max_iter], skipper)
                y_err = running_mean(R_test_info_std[i][:x_max_iter], skipper)
            else:
                y_plot = R_test_info_mean[i][:x_max_iter:skipper]
                y_err = R_test_info_std[i][:x_max_iter:skipper]

            x_ax = np.arange(0, len(y_plot)) * 10000 // 2
            if args.env_id == 21:
                x_ax = np.arange(0, len(y_plot)) * 100000 * 5 / 4

            if i == 0 and m_return_list is not None :
                ic_k = 0
                for i_k in range(0, len(m_return_list)):
                    opt_plot = (x_ax * 0) + m_return_list[i_k]
                    if i_k == 0 or i_k == 2 or i_k == 4 or i_k == 6 or i_k == 9:
                    # if i_k == 0 or i_k == 4 or i_k == 8:
                        ax_3.plot(x_ax, opt_plot, color=cc_tmp[ic_k], linestyle=":", linewidth=linewidth+0.5)
                        # plt.text(x=0, y=m_return_list[i_k], s="k=%d" % (i_k+1), fontsize=fontsize-3, color=cc_tmp[ic_k])
                        ic_k += 1

            if i != 8:    
                errorfill(x_ax, y_plot, yerr=y_err, color=info_color[i], linestyle="-", linewidth=linewidth, shade=1)
            else:
                errorfill(x_ax, y_plot, yerr=y_err, color="goldenrod", linestyle="-", linewidth=linewidth, shade=1)

        fig_name_info = "./figures/%s_info" % (plot_name)
        if plot_large: 
            fig_name_info += "_large"
        print(fig_name_info) 

        if args.env_id == 21:
            title = "%s" % ("RobosuiteReacher")
            if "infogail_best" in plot_methods:
                ax_3.set_ylim([-10, 45])    # for paper
            else:
                ax_3.set_ylim([0, 35])        # for slide 
            
        if plot_large:
            plt.title(title + " (InfoGAIL)", fontsize=fontsize+1)
        else:
            plt.title(title + " (InfoGAIL)", fontsize=fontsize)
                
        plt.xlabel("Transition samples", fontsize=fontsize-2)      
        plt.ylabel('Cumulative rewards', fontsize=fontsize-2)
        plt.xticks(fontsize=fontsize-2)
        plt.yticks(fontsize=fontsize-2)
        plt.tight_layout()
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        if args.plot_save:
            f_3.savefig(fig_name_info + ".pdf", bbox_inches='tight', pad_inches = 0)
        
    if args.plot_show:
        plt.show()

if __name__ == "__main__":
    args = args_parser()
    plot(args)
