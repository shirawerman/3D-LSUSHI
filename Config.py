import torch


class Config:

    def __init__(self, conf_json):
        for k, v in conf_json.items():
            setattr(self, k, v)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32
        self.f0 = eval(self.f0)

        self.psf_params = torch.tensor(self.psf_params, device=self.device, dtype=self.dtype)
        self.psf_variance = torch.tensor(self.psf_variance, device=self.device, dtype=self.dtype)

        self.dim_target = [self.scale_factor * i for i in self.dim_input]

        self.boundary = [round(self.scale_factor * i  / 4) for i in self.dim_input]
        grid_x, grid_y, grid_z = torch.meshgrid(
            torch.arange(-self.boundary[0], self.scale_factor * self.dim_input[0] + self.boundary[0])
            * self.dx_im / self.scale_factor,
            torch.arange(-self.boundary[1], self.scale_factor * self.dim_input[1] + self.boundary[1])
            * self.dx_im / self.scale_factor,
            torch.arange(-self.boundary[2], self.scale_factor * self.dim_input[2] + self.boundary[2])
            * self.dx_im / self.scale_factor)  # indexing='ij', m.g in metric units

        self.grid_x = grid_x.to(dtype=self.dtype, device=self.device)
        self.grid_y = grid_y.to(dtype=self.dtype, device=self.device)
        self.grid_z = grid_z.to(dtype=self.dtype, device=self.device)