import multiprocessing
import numpy as np

#this function works ONLY if foo is vectorized

def run_parallel(foo, Ncpu, args, params, return_value=True):

    ##args=(array1, array2, array3, ...)  <-- tuple, list, or array
    ##the code will run foo(array1[0], array2[0], array3[0], ..), etc

    if Ncpu=='max':
        Ncpu = multiprocessing.cpu_count()

    args   = list(args)
    params = list(params)

    Njobs = len(args[0])
    Ncpu  = min(Ncpu, Njobs)

    if Ncpu == 1:

        all_args = args + params
   
        if return_value:
            all_res = foo(*all_args)
        else:
            foo(*all_args)
        #else:   old stuff: for function not vec
        #    if return_value:
        #        for nn in range(len(params[0])):
        #            all_res.append(foo(*params[:, nn]))
        #    else:
        #        for nn in range(len(params[0])):
        #            foo(*params[:, nn])

    else :

        Njobs_cpu = int(float(Njobs)/float(Ncpu))
        
        print('.. running on ' +  str(Ncpu) + ' ('+str(Njobs_cpu)+' jobs each)')

        chunks = np.array_split(np.array(args), Ncpu, axis=1)
        #chunks = np.array_split(np.array(args), Njobs, axis=1)

        pool = multiprocessing.Pool(Ncpu)

        tasks = []
        for nn in range(Ncpu):
            tasks.append(tuple(chunks[nn]))

        results = [pool.apply_async(foo, t+tuple(params)) for t in tasks]
        
        if return_value:
            all_res = results[0].get()
            for rr in results[1:]:
                all_res = np.concatenate((all_res, rr.get()))
            
    if return_value:
        return all_res
    else:
        pass
