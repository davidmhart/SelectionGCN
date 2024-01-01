import torch
from math import sqrt, pi
import random

sqrt2 = sqrt(2)/2
sqrt3 = sqrt(3)/3

vectorList2D = torch.tensor([[1,0],[sqrt2,sqrt2],[0,1],[-sqrt2,sqrt2],[-1,0],[-sqrt2,-sqrt2],[0,-1],[sqrt2,-sqrt2]],dtype=torch.float).transpose(1,0)
vectorList3D = torch.tensor([[sqrt2,0,-sqrt2],[sqrt3,sqrt3,-sqrt3],[0,sqrt2,-sqrt2],[-sqrt3,sqrt3,-sqrt3],[-sqrt2,0,-sqrt2],[-sqrt3,-sqrt3,-sqrt3],[0,-sqrt2,-sqrt2],[sqrt3,-sqrt3,-sqrt3],
                            [1,0,0],[sqrt2,sqrt2,0],[0,1,0],[-sqrt2,sqrt2,0],[-1,0,0],[-sqrt2,-sqrt2,0],[0,-1,0],[sqrt2,-sqrt2,0],
                            [sqrt2,0,sqrt2],[sqrt3,sqrt3,sqrt3],[0,sqrt2,sqrt2],[-sqrt3,sqrt3,sqrt3],[-sqrt2,0,sqrt2],[-sqrt3,-sqrt3,sqrt3],[0,-sqrt2,sqrt2],[sqrt3,-sqrt3,sqrt3]],dtype=torch.float).transpose(1,0)

def direction_selection(num_dirs):
    # Generate equally spaced unit vectors
    radians = 2*pi*torch.arange(num_dirs)/(num_dirs)
    x_vals = torch.sin(radians) # Start from the vector (0,1)
    y_vals = torch.cos(radians)
    vectorList = torch.vstack((x_vals,y_vals))
    
    # Curry with the max direction function
    def f(x, edge_index):
        return max_direction(x,edge_index,vectorList)
    
    return f
    
def direction3D_selection(num_dirs):

    if num_dirs == 6:
        f = max_dimension        
    elif num_dirs == 26:
        # Curry with the max direction function
        def f(x, edge_index):
            return max_direction(x,edge_index,vectorList3D)
    else:
        raise ValueError("General number of directions in 3D not currently supported")
        
    return f
    
def distance_selection(num_dists):

    # Curry with the distance function
    def f(x, edge_index):
        distance = torch.sqrt(torch.sum(torch.square(x[edge_index[1]] - x[edge_index[0]]),dim=1))    
        max_distance = int(torch.std(distance,dim=0,unbiased=True))
        
        return distance_cutoff(distance,num_dists,max_distance)

    return f
    
def direction_distance_selection(num_dirs,num_dists,use3D=False):

    if use3D:
        direction_function = direction3D_selection(num_dirs)
    else:
        direction_function = direction_selection(num_dirs)

    distance_function = distance_selection(num_dists)
    
    # Curry combination function
    def f(x, edge_index):
        selections_dir = direction_function(x,edge_index)
        selections_dist = distance_function(x,edge_index)
        return combine_selections(selections_dir,selections_dist)

    # Calculate the new selection count of the combined function
    selection_count = num_dirs * (num_dists - 1) + 1

    return f, selection_count

def attribute_selection(start_index,end_index,pairwise=False):
    
    if pairwise:
        def f(x,edge_index):
            selections = one_hot_comparison(x[:,start_index:end_index], edge_index)
            return selections
    else:
        def f(x,edge_index):
            selections = one_hot_conversion(x[:,start_index:end_index], edge_index)
            return selections
    
    return f



def max_dimension(x,edge_index):
    '''Make selections by taking the maximum feature dimension in positive and negative directions'''

    directions_pos = x[edge_index[1]] - x[edge_index[0]]
    
    # Append the negative of the directions to simplify the max function
    directions_neg = -directions_pos
    directions = torch.cat((directions_pos,directions_neg),dim=1)

    # Find the maximum feature to make the selection
    selections = torch.argmax(directions,dim=1) + 1
    
    # Make the zero selection if the edge is a self loop
    selections[torch.where(edge_index[0] == edge_index[1])] = 0
    
    return selections

def max_direction(x,edge_index,vectorList=vectorList2D):
    '''Make selections by matching to the closest direction'''
        
    directions = x[edge_index[1]] - x[edge_index[0]]

    # Take the dot product between each edge and the possible directions
    selections = torch.argmax(torch.matmul(directions,vectorList.to(directions.device)),dim=1) + 1

    #del vectorList
    
    # Make the zero selection if the edge is a self loop
    selections[torch.where(edge_index[0] == edge_index[1])] = 0
    
    return selections

def distance_binning(distances,bins=4,log=False,separate_max_bin=True):
    '''Assumes distances is given as a number between 0 and 1, 1 being the closest'''
    if log:
        distances = -torch.log(distances)
        distances = torch.minimum(distances,torch.tensor(1e4))
        distances = 1 - distances/torch.amax(distances)

    if separate_max_bin:
        selections = torch.floor((distances)*(bins-1))
    else:
        selections = torch.floor((distances)*bins)
        selections = torch.minimum(selections,torch.tensor(bins-1,device=selections.device))
    return selections

def distance_cutoff(distances,bins=4,max_distance=0.1,epsilon=1e-3):
    
    selections = torch.ceil((distances/max_distance)*(bins-1))
    
    # Cutoff for nodes furter than max distance
    selections = torch.minimum(selections,torch.tensor(bins-1,device=selections.device))
    
    # Cutoff for nodes very close to the center
    selections[torch.where(distances<max_distance*epsilon)] = 0
    
    return selections

 
def one_hot_conversion(x,edge_index):
    '''Assumes binary features'''
    target = x[edge_index[1]]
    selections = torch.argmax(target,dim=1)
    return selections

def one_hot_comparison(x,edge_index,allow_nulls=False):
    '''Assumes binary features'''
    source = x[edge_index[0]]
    target = x[edge_index[1]]
        
    source_index = torch.argmax(source,dim=1)
    target_index = torch.argmax(target,dim=1)
    
    # A selection from every possible one-hot to every other
    selections = (torch.amax(target_index)+1)*source_index + target_index
    
    return selections.long()
   
def combine_selections(selections1,selections2,single_zero=True):
    
    if single_zero:
        # Assumes the zeroth selection is unique and designates a connection to the same node
        # Don't want to combine with other selecitons
        selections1 = selections1 - 1
        selections2 = selections2 - 1
        selections = selections1 + (torch.amax(selections1)+1)*selections2 + 1
        selections[torch.where(selections1==-1)] = 0
        selections[torch.where(selections2==-1)] = 0
    else:
        selections = selections1 + (torch.amax(selections1)+1)*selections2
        
    return selections
   
def shared_count(x,edge_index,return_differences=False,bins=4):
    '''Assumes binary features'''
    bool_x = x.bool()

    shared = torch.logical_and(bool_x[edge_index[1]],bool_x[edge_index[0]])

    count = torch.sum(shared,dim=1)

    #print(shared.shape)
    #print(count.shape)

    if bins > 0:
        max_count = torch.max(count)
        count = torch.floor((count/(max_count+1))*(bins))

    #print(torch.amin(count),torch.amax(count))
        
    if return_differences:
        diffs = torch.logical_xor(bool_x[edge_index[1]],bool_x[edge_index[0]])

        if bins>0:
            max_diffs = torch.max(diffs)
            diffs = torch.floor((diff/(max_diffs+1))*(bins))

        return torch.stack((count,-torch.sum(diffs,dim=1)),dim=1)

    else:
        return count

def shared_attributes(x,edge_index,indices=None):
    '''Assumes binary features'''
    bool_x = x.bool()

    if indices is None:
        indices = [random.randint(0,x.shape[1]-1)]

    selections = []
    for index in indices:
        cur_x = bool_x[:,index]

        source = cur_x[edge_index[0]]
        target = cur_x[edge_index[1]]
        
        cur_selections = torch.zeros(len(source),dtype=torch.long,device=x.device)

        cur_selections[torch.where(torch.logical_and(source,torch.logical_not(target)))] = 1
        cur_selections[torch.where(torch.logical_and(torch.logical_not(source),target))] = 2
        cur_selections[torch.where(torch.logical_and(source,target))] = 3

        selections.append(cur_selections)

    selections = torch.stack(selections,dim=-1)
    
    selections = selections.squeeze() # Handle single selections

    return selections 

def compare_attributes(x,edge_index,indices=None):
    if indices is None:
        indices = [random.randint(0,x.shape[1]-1)]
        
    selections = []
    for index in indices:
        cur_x = x[:,index]

        source = cur_x[edge_index[0]]
        target = cur_x[edge_index[1]]
        
        cur_selections = torch.zeros(len(source),dtype=torch.long,device=x.device) 
        cur_selections[torch.where(target > source)] = 1
        cur_selections[torch.where(target < source)] = 2

        selections.append(cur_selections)

    selections = torch.stack(selections,dim=-1)
    
    selections = selections.squeeze() # Handle single selections

    return selections 

def on_road(locations,edge_index):
    
    source = edge_index[0]
    target = edge_index[1]
    
    selections = torch.zeros(len(source),dtype=torch.long)
    
    for i in range(locations.shape[0]):
        
        indices = torch.where(source==i)
        
        point_set = target[indices]
        my_locations = locations[point_set]
        center = torch.mean(my_locations,dim=0)
        my_centered = my_locations - center
        
        # Find best fit line using least_square
        A = torch.cat((my_centered,torch.ones((1,2),dtype=torch.float)),dim=0)
        b = torch.cat((torch.zeros((len(point_set),1),dtype=torch.float),torch.ones((1,1),dtype=torch.float)),dim=0)
        
        try:
            # Least Squares
            inverse = torch.linalg.inv(torch.matmul(torch.t(A),A))
            result = torch.matmul(torch.matmul(inverse,torch.t(A)),b)

            x_diff = -result.data[0]
            y_diff = result.data[1]

            # Compare each point to determine which direction of flow they are in
            cur_selections = (my_centered[:,0]*x_diff + my_centered[:,1]*y_diff > 0).long()

            selections[indices] = cur_selections
        except:
            print("Singular matrix detected")
        
    return selections
        
