from typing import List, Optional, Union, Type, Tuple, Literal
import torch
from torch import nn
from xbarray.transformations.pointcloud.pytorch import ball_query, farthest_point_sampling


@torch.compiler.disable
def sample_farthest_points(
    points : torch.Tensor,
    lengths : torch.Tensor,
    K : int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Farthest point sampling of the first ``lengths`` points of each cloud, starting at index 0.

    Not compiled: the loop has ``K`` iterations, which ``torch.compile`` would unroll.

    Returns ``(N, K, D)`` points (zeros where invalid) and the ``(N, K)`` valid mask. A
    cloud yields ``min(K, length)`` valid samples, and always its first point, even
    when it is empty.
    """
    N, P, D = points.shape
    with torch.no_grad():
        slots = torch.arange(P, device=points.device).unsqueeze(0)
        _, idx, _ = farthest_point_sampling(points.float(), K, None, points_valid=slots < lengths.unsqueeze(-1))
        valid = torch.arange(K, device=points.device).unsqueeze(0) < lengths.clamp(min=1).unsqueeze(-1)
    sampled = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, D))
    return sampled.masked_fill(~valid.unsqueeze(-1), 0.0), valid


@torch.compiler.disable
def query_ball_neighbors(
    query_points : torch.Tensor,
    query_valid : torch.Tensor,
    points : torch.Tensor,
    lengths : torch.Tensor,
    K : int,
    radius : float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``xbarray``'s ``ball_query`` on prefix-valid clouds: ``(N, Q, K)`` neighbor indices and valid mask."""
    with torch.no_grad():
        slots = torch.arange(points.shape[1], device=points.device).unsqueeze(0)
        idx, _, valid = ball_query(
            query_points, points, K, radius,
            query_valid=query_valid,
            points_valid=slots < lengths.unsqueeze(-1),
        )
    return idx, valid


def gather_neighbors(values : torch.Tensor, idx : torch.Tensor, valid : torch.Tensor) -> torch.Tensor:
    """Gather ``(N, P, D)`` values at ``(N, Q, K)`` indices into ``(N, Q, K, D)``, zeros where invalid."""
    N, Q, K = idx.shape
    gathered = torch.gather(
        values.unsqueeze(1).expand(-1, Q, -1, -1), 2,
        idx.unsqueeze(-1).expand(-1, -1, -1, values.shape[-1]),
    )
    return gathered.masked_fill(~valid.unsqueeze(-1), 0.0)


class PointNetIndividualLayer(nn.Module):
    def __init__(
        self,
        d_input,
        d_output,
        norm_layer=nn.LayerNorm,
        act_layer=nn.ReLU
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_input, d_output),
            norm_layer(d_output),
            act_layer()
        )
    
    def forward(self, x):
        return self.mlp(x)

class DP3IndividualEncoder(nn.Module):
    def __init__(
        self,
        d_input = 3,
        d_intermediate = [64, 128, 256],
        norm_layer = nn.Identity,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_intermediate[-1]

        layers = []
        dimensions = [d_input] + d_intermediate
        for i in range(len(dimensions) - 1):
            layers.append(PointNetIndividualLayer(
                dimensions[i], 
                dimensions[i+1], 
                norm_layer=norm_layer if i < len(dimensions) - 2 else nn.Identity,
                act_layer=nn.ReLU if i < len(dimensions) - 2 else nn.Identity,
            ))
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x : torch.Tensor):
        """
        Input:
            x: (..., d_input) where B is batch size, N is number of points, d_input is input dimension
        Output:
            y: (..., d_output)
        """
        x = self.encoder(x) # (..., d_output)
        return x

class DP3Reduction(nn.Module):
    def __init__(
        self,
        method : Literal['max', 'mean'] = 'max',
        dim : int = -2
    ):
        super().__init__()
        self.method = method
        self.dim = dim
    
    def forward(self, x : torch.Tensor, x_valid : Optional[torch.Tensor] = None):
        """
        Input:
            x: (..., N, d) where B is batch size, N is number of points, d is feature dimension
            x_valid: (..., N) where True indicates valid points and False indicates invalid points. It has to be that the first N' items are valid and the rest are invalid.
        Output:
            y: (..., d)
            y_valid: optional (...) where True indicates the reduced feature is valid (i.e. there is at least one valid point in the reduced dimension) and False indicates the reduced feature is invalid (i.e. there is no valid point in the reduced dimension). This is only returned when x_valid is not None.
        """
        if self.method == 'mean':
            if x_valid is not None:
                x = torch.where(x_valid.unsqueeze(-1), x, 0.0)
                summed = x.sum(dim=self.dim)
                counts = x_valid.unsqueeze(-1).sum(dim=self.dim).clamp(min=1)
                reduced = summed / counts
            else:
                reduced = x.mean(dim=self.dim)
        else:
            if x_valid is not None:
                x = torch.where(
                    x_valid.unsqueeze(-1), x, -torch.inf
                )
            
            # max(), not amax(): same values, but its backward routes the gradient
            # through the argmax indices instead of a full-size equality mask,
            # which makes the backward pass noticeably cheaper.
            reduced = x.max(dim=self.dim).values
        
        if x_valid is not None:
            reduced_valid = x_valid.unsqueeze(-1).any(dim=self.dim).squeeze(-1)
            return reduced, reduced_valid

        return reduced, None

class DP3Projection(nn.Module):
    def __init__(
        self,
        d_input : int = 256,
        d_output : int = 64,
        norm_layer : Type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.linear = nn.Linear(d_input, d_output)
        self.norm = norm_layer(d_output)
        self.d_input = d_input
        self.d_output = d_output
    
    def forward(self, x : torch.Tensor):
        x = self.linear(x)
        y = self.norm(x)
        return y

class DP3Encoder(nn.Module):
    def __init__(
        self,
        d_input : int = 3,
        d_individual_intermediate = [64, 128],
        d_individual_output : Optional[int] = 256,
        d_output : int = 64,
        dim_reduce : int = -2,
        reduce_method : Literal['max', 'mean'] = 'max',
        norm_layer : Type[nn.Module] = nn.LayerNorm,
        last_norm : bool = True,
        enable_cls_token : bool = True
    ):
        super().__init__()
        if d_individual_output is not None:
            d_individual_intermediate = [*d_individual_intermediate, d_individual_output]
        self.encoder = DP3IndividualEncoder(
            d_input, 
            d_individual_intermediate, 
            norm_layer=norm_layer
        )
        self.projection = DP3Projection(self.encoder.d_output, d_output, norm_layer=norm_layer if last_norm else nn.Identity)
        self.last_norm = last_norm
        self.reduction = DP3Reduction(reduce_method, dim_reduce)

        if enable_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(self.encoder.d_output))
            self.cls_token = nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None
    
    @property
    def d_input(self):
        return self.encoder.d_input
    
    @property
    def d_output(self):
        return self.projection.d_output

    def forward(self, x : torch.Tensor, x_valid : Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Input:
            x: (B, N, d_input) or (B, N, d_input+1)
            x_valid: (B, N)
        Output:
            y: (B, d_output)
            individual_features: (B, N, d_individual_output)
        """
        if x.shape[-1] == self.encoder.d_input + 1:
            assert x_valid is None
            x_valid = x[..., -1] != 0
            x = x[..., :-1]
        else:
            assert x_valid is None or x_valid.shape == x.shape[:-1]
            assert x.shape[-1] == self.encoder.d_input
    
        features = self.encoder(x) # (B, N, d_individual_output)
        features_reduced, features_reduced_valid = self.reduction(features, x_valid) # (B, d_individual_output), (B, )
        
        if features_reduced_valid is not None:
            features_reduced = torch.where(
                features_reduced_valid.unsqueeze(-1),
                features_reduced,
                self.cls_token.reshape([1] * (features_reduced.ndim - 1) + [features_reduced.shape[-1]]) if self.cls_token is not None else 0.0
            )
        
        out_features = self.projection(features_reduced) # (B, d_output)
        return out_features, features_reduced_valid

class IDP3Encoder(nn.Module):
    def __init__(
        self,
        d_input : int = 3,
        d_individual_intermediate = [64, 128, 256],
        d_output : int = 64,
        dim_reduce : int = -2,
        reduce_method : Literal['max', 'mean'] = 'max',
        norm_layer : Type[nn.Module] = nn.Identity,
        last_norm : bool = True,
        enable_cls_token : bool = True
    ):
        super().__init__()

        self._d_input = d_input
        layers = []
        global_layers = []
        dimensions = [d_input] + d_individual_intermediate
        for i in range(len(dimensions) - 1):
            layers.append(PointNetIndividualLayer(
                dimensions[i],
                dimensions[i+1], 
                norm_layer=norm_layer if i < len(dimensions) - 2 else nn.Identity,
                act_layer=nn.LeakyReLU if i < len(dimensions) - 2 else nn.Identity,
            ))
            global_layers.append(PointNetIndividualLayer(
                dimensions[i+1] * 2,
                dimensions[i+1],
                norm_layer=nn.Identity,
                act_layer=nn.LeakyReLU
            ))
        self.encoder = nn.Sequential(*layers)
        self.global_encoder = nn.Sequential(*global_layers)
        self.projection = DP3Projection(d_individual_intermediate[-1], d_output, norm_layer=norm_layer if last_norm else nn.Identity)
        self.last_norm = last_norm
        self.reduction = DP3Reduction(reduce_method, dim_reduce)

        if enable_cls_token:
            self.cls_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(dim)) for dim in dimensions[1:]
            ])
            self.final_cls_token = nn.Parameter(torch.zeros(dimensions[-1]))
            for cls_token in self.cls_tokens:
                nn.init.trunc_normal_(cls_token, std=0.02)
            nn.init.trunc_normal_(self.final_cls_token, std=0.02)
        else:
            self.cls_tokens = None
            self.final_cls_token = None
    
    @property
    def d_input(self):
        return self._d_input
    
    @property
    def d_output(self):
        return self.projection.d_output

    def forward(self, x : torch.Tensor, x_valid : Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]] :
        """
        Input:
            x: (B, N, d_input) or (B, N, d_input+1)
            x_valid: (B, N)
        Output:
            y: (B, N, d_output)
        """
        if x.shape[-1] == self.d_input + 1:
            assert x_valid is None
            x_valid = x[..., -1] != 0
            x = x[..., :-1]
        else:
            assert x_valid is None or x_valid.shape == x.shape[:-1]
            assert x.shape[-1] == self.d_input
        
        for i, (layer, global_layer) in enumerate(zip(self.encoder, self.global_encoder)):
            cls_token = self.cls_tokens[i] if self.cls_tokens is not None else None
            x = layer(x)
            reduced, reduced_valid = self.reduction.forward(x, x_valid)
            if reduced_valid is not None:
                reduced = torch.where(
                    reduced_valid.unsqueeze(-1),
                    reduced,
                    cls_token.reshape([1] * (reduced.ndim - 1) + [reduced.shape[-1]]) if cls_token is not None else 0.0
                )
            x = torch.concat([x, reduced.unsqueeze(self.reduction.dim).expand_as(x)], dim=-1)
            x = global_layer(x)
        
        reduced, reduced_valid = self.reduction.forward(x, x_valid)
        if reduced_valid is not None:
            reduced = torch.where(
                reduced_valid.unsqueeze(-1),
                reduced,
                self.final_cls_token.reshape([1] * (reduced.ndim - 1) + [reduced.shape[-1]]) if self.final_cls_token is not None else 0.0
            )
        out_features = self.projection(reduced)
        return out_features, reduced_valid

class PointNet2SetAbstraction(nn.Module):
    def __init__(
        self,
        group_num : int,
        d_input : int = 3,
        d_individual_intermediate = [64, 128],
        d_individual_output : int = 256,
        d_output : int = 64,
        group_max_points : int = 32,
        group_radius : float = 0.1,
        reduce_method : Literal['max', 'mean'] = 'max',
        norm_layer : Type[nn.Module] = nn.LayerNorm,
        enable_cls_token : bool = True,
    ):
        super().__init__()
        self.set_dp3 = DP3Encoder(
            3 + d_input,
            d_individual_intermediate,
            d_individual_output,
            d_output,
            dim_reduce=-2,
            reduce_method=reduce_method,
            norm_layer=norm_layer,
            enable_cls_token=enable_cls_token,
        )
        self.group_num = group_num
        self.group_radius = group_radius
        self.group_max_points = group_max_points
        assert group_radius > 0, "group_radius must be positive"
        
        self.reduce_method = reduce_method
    
    @property
    def d_input(self):
        return self.set_dp3.d_input - 3
    
    @property
    def d_output(self):
        return self.set_dp3.d_output
    
    def forward(self, pos_x : torch.Tensor, features_x : Optional[torch.Tensor], mask : Optional[torch.Tensor] = None, input_fps_ordered : bool = False):
        """
        Input:
            pos_x: (B, N, 3) or (B, N, 4) for valid mask
            features_x: (B, N, d_input) or None if d_input is 3 (use the pointcloud)
            mask: (B, N) or None => It has to be that the first N' items are valid and the rest are invalid
            input_fps_ordered: pos_x is already in farthest point sampling order (a previous layer's
                queries), so the first group_num points are this layer's queries: greedy sampling of
                an FPS-ordered set from its first point returns that same order.
        Output:
            query_pos_x: (B, group_num, 3)
            grouped_features: (B, group_num, d_output)
            query_valids: (B, group_num)
        """
        B, N, _ = pos_x.shape
        if pos_x.shape[-1] == 4:
            assert mask is None
            mask = pos_x[..., -1] != 0 # (B, N)
            pos_x = pos_x[..., :-1] # (B, N, d_input)
        else:
            assert mask is None or mask.shape == pos_x.shape[:-1]
            assert pos_x.shape[-1] == 3

        pos_x = pos_x.float()

        if features_x is not None:
            assert features_x.shape == (B, N, self.d_input)
        else:
            assert self.d_input == 3
            features_x = pos_x

        if mask is not None:
            lengths_x = torch.sum(mask, dim=-1).long() # (B,)
        else:
            lengths_x = torch.full((B,), N, dtype=torch.int64, device=pos_x.device)

        if input_fps_ordered:
            query_pos_x = pos_x[:, :self.group_num] # (B, group_num, 3)
            query_valids = mask[:, :self.group_num] # (B, group_num)
        else:
            query_pos_x, query_valids = sample_farthest_points(pos_x, lengths_x, self.group_num) # (B, group_num, 3), (B, group_num)
        neighbor_idx, grouped_valid = query_ball_neighbors(
            query_pos_x, query_valids, pos_x, lengths_x, self.group_max_points, self.group_radius
        ) # (B, group_num, group_max_points) each
        grouped_pos = gather_neighbors(pos_x, neighbor_idx, grouped_valid) # (B, group_num, group_max_points, 3)
        grouped_features = gather_neighbors(features_x.float(), neighbor_idx, grouped_valid) # (B, group_num, group_max_points, d_input)
        normalized_grouped_pos = (grouped_pos - query_pos_x.unsqueeze(-2)) / self.group_radius

        # Now every group should go through the DP3Encoder
        grouped_features, _ = self.set_dp3.forward(
            torch.concat([
                normalized_grouped_pos,
                grouped_features
            ], dim=-1) if grouped_features is not None else normalized_grouped_pos,
            x_valid=grouped_valid
        ) # (B, group_num, d_output)
        return query_pos_x, grouped_features, query_valids

class PointNet2Encoder(nn.Module):
    def __init__(
        self,
        d_input : int = 3,
        sa_intermediate_hidden_dim_start : List[int] = [32, 64, 128],
        sa_d_outputs : List[int] = [32, 64, 128],
        sa_group_nums : List[int] = [128, 32, 8],
        sa_group_max_points : List[int] = [64, 8, 8],
        sa_group_radius : Union[List[float], float] = [0.1, 0.3, 0.5],
        reduce_method : Literal['max', 'mean'] = 'max',
        norm_layer : Type[nn.Module] = nn.LayerNorm,
        enable_cls_token_layers : bool = False,
        enable_cls_token : bool = True,
    ):
        super().__init__()
        last_layer_d_output = d_input
        sa_layers = []
        for i, hidden_dim_start, d_output, group_num, max_points in zip(
            range(len(sa_intermediate_hidden_dim_start)),
            sa_intermediate_hidden_dim_start,
            sa_d_outputs,
            sa_group_nums,
            sa_group_max_points
        ):
            sa_layers.append(
                PointNet2SetAbstraction(
                    group_num,
                    last_layer_d_output,
                    [hidden_dim_start, hidden_dim_start*2],
                    hidden_dim_start*4,
                    d_output,
                    group_max_points=max_points,
                    group_radius=sa_group_radius[i] if isinstance(sa_group_radius, list) else sa_group_radius,
                    reduce_method=reduce_method,
                    norm_layer=norm_layer,
                    enable_cls_token=enable_cls_token_layers,
                )
            )
            last_layer_d_output = d_output
        self.sa_layers = nn.ModuleList(sa_layers)
        self.d_output = last_layer_d_output
        if enable_cls_token:
            self.cls_token = None if not enable_cls_token else nn.Parameter(torch.zeros(last_layer_d_output))
            self.cls_pos = None if not enable_cls_token else nn.Parameter(torch.zeros(3))
            self.cls_token = nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.cls_pos = nn.init.trunc_normal_(self.cls_pos, std=0.02)
        else:
            self.cls_token = None
            self.cls_pos = None
    
    def forward(
        self,
        pos_x : torch.Tensor,
        features_x : Optional[torch.Tensor] = None,
        mask : Optional[torch.Tensor] = None
    ):
        """
        Input:
            pos_x: (B, N, 3) or (B, N, 4) for valid mask
            features_x: (B, N, d_input) or None if d_input is 3 (use the pointcloud)
            mask: (B, N) or None => It has to be that the first N' items are valid and the rest are invalid
        Output:
            y: (B, N_final_groups, d_output)
            pos_y: (B, N_final_groups, 3)
            mask_y: (B, N_final_groups)
        """
        for i, sa_layer in enumerate(self.sa_layers):
            pos_x, features_x, mask = sa_layer(pos_x, features_x, mask=mask, input_fps_ordered=i > 0)
        
        if self.cls_token is not None:
            expanded_cls_token = self.cls_token[None, None, :].expand(
                pos_x.shape[0], 1, -1
            )
            expanded_cls_pos = self.cls_pos[None, None, :].expand(
                pos_x.shape[0], 1, -1
            )
            mask = None if mask is None else nn.functional.pad(mask, (1, 0), value=True)
            pos_x = torch.concat([
                expanded_cls_pos,
                pos_x
            ], dim=1)
            features_x = torch.concat([
                expanded_cls_token,
                features_x
            ], dim=1)

        return features_x, pos_x, mask
