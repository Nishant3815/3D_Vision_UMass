import torch
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU, GroupNorm

# the "MLP" block that you will use the in the PointNet and CorrNet modules you will implement
# This block is made of a linear transformation (FC layer), 
# followed by a Leaky RelU, a Group Normalization (optional, depending on enable_group_norm)
# the Group Normalization (see Wu and He, "Group Normalization", ECCV 2018) creates groups of 32 channels
def MLP(channels, enable_group_norm=True):
    if enable_group_norm:
        num_groups = [0]
        for i in range(1, len(channels)):
            if channels[i] >= 32:
                num_groups.append(channels[i]//32)
            else:
                num_groups.append(1)    
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(negative_slope=0.2), GroupNorm(num_groups[i], channels[i]))
                     for i in range(1, len(channels))])
    else:
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(negative_slope=0.2))
                     for i in range(1, len(channels))])


# PointNet module for extracting point descriptors
# num_input_features: number of input raw per-point or per-vertex features 
# 		 			  (should be 3, since we have 3D point positions in this assignment)
# num_output_features: number of output per-point descriptors (should be 32 for this assignment)
# this module should include
# - a MLP that processes each point i into a 128-dimensional vector f_i
# - another MLP that further processes these 128-dimensional vectors into h_i (same number of dimensions)
# - a max-pooling layer that collapses all point features h_i into a global shape representaton g
# - a concat operation that concatenates (f_i, g) to create a new per-point descriptor that stores local+global information
# - a MLP that transforms this concatenated descriptor into the output 32-dimensional descriptor x_i
# **** YOU SHOULD CHANGE THIS MODULE, CURRENTLY IT IS INCORRECT ****
class PointNet(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(PointNet, self).__init__()
        self.mlp          = MLP([num_input_features,32,64,128])
        self.mlp2         = MLP([128,128])
        self.mlp3         = MLP([256,128,64])
        self.linear_layer = Lin(64,num_output_features)

    def forward(self, x):
        batch_size  = x.size()[0] # Represents number of points/vertex
        x1          = self.mlp(x)
        x2          = self.mlp2(x1)
        # Take the maximum values to replicate max pooling and repeat the vector batch_size times for proper concatenation
        x3,_        = torch.max(x2,0)
        x3          = x3.view(1,len(torch.max(x2,0).values)).repeat(batch_size,1)
        x3          = torch.cat([x3,x1],dim=1)
        x3          = self.mlp3(x3)
        x3          = self.linear_layer(x3)
        #print("Shape of x3 is: ",x3.size())
        return x3


# CorrNet module that serves 2 purposes:  
# (a) uses the PointNet module to extract the per-point descriptors of the point cloud (out_pts)
#     and the same PointNet module to extract the per-vertex descriptors of the mesh (out_vtx)
# (b) if self.train_corrmask=1, it outputs a correspondence mask
# The CorrNet module should
# - include a (shared) PointNet to extract the per-point and per-vertex descriptors 
# - normalize these descriptors to have length one
# - when train_corrmask=1, it should include a MLP that outputs a confidence 
#   that represents whether the mesh vertex i has a correspondence or not
#   Specifically, you should use the cosine similarity to compute a similarity matrix NxM where
#   N is the number of mesh vertices, M is the number of points in the point cloud
#   Each entry encodes the similarity of vertex i with point j
#   Use the similarity matrix to find for each mesh vertex i, its most similar point n[i] in the point cloud 
#   Form a descriptor matrix X = NxF whose each row stores the point descriptor of n[i] (from the point cloud descriptors)
#   Form a vector S = Nx1 whose each entry stores the similarity of the pair (i, n[i])
#   From the PointNet, you also have the descriptor matrix Y = NxF storing the per-vertex descriptors
#   Concatenate [X Y S] into a N x (2F + 1) matrix
#   Transform this matrix into the correspondence mask Nx1 through a MLP followed by a linear transformation
# **** YOU SHOULD CHANGE THIS MODULE, CURRENTLY IT IS INCORRECT ****
class CorrNet(torch.nn.Module):
    def __init__(self, num_output_features, train_corrmask):        
        super(CorrNet, self).__init__()
        self.train_corrmask = train_corrmask
        self.num_req_concat_feat = 2*num_output_features+1
        #print("2F+1:", self.num_req_concat_feat)
        self.pointnet_share = PointNet(3, num_output_features)
        self.mlp_final = MLP([self.num_req_concat_feat, 64])
        self.linear_final = Lin(64,1)
        
    def forward(self, vtx, pts):
        out_vtx = self.pointnet_share(vtx)
        out_pts = self.pointnet_share(pts)
        #print("Size of out_vtx after pointnet is: ", out_vtx.size())
        #print("Size of out_pts after pointnet is: ", out_pts.size())
        # Normalize the vertex
        out_vtx_norm = out_vtx.div(out_vtx.norm(p=2,dim=1,keepdim=True).expand_as(out_vtx))
        # Normalize the points 
        out_pts_norm = out_pts.div(out_pts.norm(p=2,dim=1,keepdim=True).expand_as(out_pts))
        #print(out_vtx_norm[0,:])
        if self.train_corrmask: 
           # print("Entered train_corrmask is True flag")
           sim_mat = torch.mm(out_vtx_norm, out_pts_norm.T)            
           corr_point_feat = out_pts_norm[torch.argmax(sim_mat,dim=1)] 
           max_sim_vector  = torch.max(sim_mat,dim=1).values.view(sim_mat.size()[0],1)
           #print(corr_point_feat.size(),out_vtx_norm.size(),max_sim_vector.size())
           #print(out_vtx_norm.size())
           concat_feats = torch.cat([out_vtx_norm,corr_point_feat,max_sim_vector],dim=1)
           #print(concat_feats.size())
           out          = self.mlp_final(concat_feats)
           out_corrmask = self.linear_final(out)     
           #print("Output values for corresponding mask output",out_corrmask.size())
        else:
            return out_vtx_norm,out_pts_norm, None

        return out_vtx_norm, out_pts_norm, out_corrmask