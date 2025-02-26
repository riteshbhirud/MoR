import json
path = "/home/yongjia/dgl/Yongjia/MOE_20250222/Planning/data/finetune/prime/1000.json"
with open(path, 'r') as f:
    data = json.load(f)
    
def combine_triplets_by_boundary_no_loops(NTar):
    combined = NTar[:]
    changes = True  # Track if merges happen during an iteration

    while changes:
        changes = False  # Reset changes for this iteration
        new_combined = []  # To hold the merged triplets
        used = [False] * len(combined)  # Track which triplets have been processed

        for i in range(len(combined)):
            if used[i]:
                continue  # Skip already processed triplets
            
            current_triplet = combined[i]  # Convert triplet to a mutable list
            used[i] = True
            merged = False  # Track if the current triplet is merged with another

            for j in range(len(combined)):
                if i != j and not used[j]:
                    other_triplet = combined[j]

                    # Check if the current triplet can be merged with another
                    if current_triplet[-1] == other_triplet[0]:  # Current's last matches other's first
                        current_triplet.extend(other_triplet[1:])  # Merge, excluding duplicate entity
                        used[j] = True
                        merged = True
                        changes = True  # A merge occurred
                        break
                    elif current_triplet[0] == other_triplet[-1]:  # Current's first matches other's last
                        current_triplet = other_triplet[:-1] + current_triplet  # Merge, excluding duplicate entity
                        used[j] = True
                        merged = True
                        changes = True
                        break

            # After merging or if no merge happened, add the triplet to the new list
            new_combined.append(current_triplet)

        # Update the combined list with the newly merged triplets
        combined = new_combined

    return combined


def combine_tar_ntar(tar, ntar):
    for nt in ntar:
        for i in range(len(tar)):
            if tar[i][-1] == nt[0]:
                tar[i] = tar[i]+nt[1:]
            elif tar[i][0] == nt[-1]:
                tar[i] = nt[:-1]+tar[i]
    return tar

def check_order(routes, target):
    for i in range(len(routes)):
        if routes[i][-1] != target:
            if routes[i][0] != target:
                raise ValueError(f"Wrong order: {routes[i]}")
            else:
                routes[i] = routes[i][::-1]
    return routes

routes_list = []
restrictions_list = []
for i in range(len(data)):
    triplets = data[i]['Triplets']
    target = data[i]['Target']
    restrictions = data[i]['Restriction']
    tar = []
    ntar = []
    for tp in triplets:
        if target in tp:
            tar.append(tp)
        else:
            ntar.append(tp)
    if len(ntar) > 0:
        ntar = combine_triplets_by_boundary_no_loops(ntar)
        routes = combine_tar_ntar(tar, ntar)
    else:
        routes = tar
    print(target)
    routes = check_order(routes, target)
    routes_list.append(routes)
    restrictions_list.append(restrictions)
    
