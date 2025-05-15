def group_rocs(automasks):
    '''
    Perform ROC grouping in a list of automasked ROCs.
    Input:
    - automasks: list of lists of the form [ladder, module, ROC]
    Returns:
    - list of lists of the form [ladder, module, first ROC, last ROC]
      where ROCs in the same ladder and module have been grouped together.
    Note: there can still be multiple entries per ladder/module, if there are disjoint ROC sets,
          e.g. [[1, 1, 0, 3], [1, 1, 8, 11]]
    '''
    
    # make a grouping of ROCs per ladder/module
    grouping = {}
    for mask in automasks:
        lmcoords = (mask[0], mask[1])
        if lmcoords not in grouping: grouping[lmcoords] = [mask[2]]
        else: grouping[lmcoords].append(mask[2])
            
    # for each ladder/module, group ROCs as much as possible
    res = []
    for lmcoords, rocs in grouping.items():
        rocs = sorted(rocs)
        inds = [0] + [ind for ind, (i,j) in enumerate(zip(rocs, rocs[1:]), 1) if j-i>1] + [len(rocs)+1]
        rocs = [rocs[i:j] for i,j in zip(inds, inds[1:])]
        for subset in rocs: res.append([lmcoords[0], lmcoords[1], subset[0], subset[-1]])
    return res
           
    
def simplify_automask(automasks):
    '''
    Perform ROC re-grouping in a list of automasked ROCs.
    Input:
    - automasks: list of lists of the form [ladder, module, first ROC, last ROC]
    Returns:
    - list of lists of the form [ladder, module, first ROC, last ROC]
      where neighbouring sets of ROCs in the same ladder and module have been grouped together.
    Note: there can still be multiple entries per ladder/module, if there are disjoint ROC sets,
          e.g. [[1, 1, 0, 3], [1, 1, 8, 11]]
    '''
    
    # expand
    automasks_expanded = []
    for automask in automasks:
        for roc in range(automask[2], automask[3]+1):
            automasks_expanded.append([automask[0], automask[1], roc])
    
    # contract
    return group_rocs(automasks_expanded)