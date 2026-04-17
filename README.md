# Path Tracer
A physically-based renderer written in Metal and Swift. The renderer leverages Metal's ray tracing capabilities for GPU acceleration. 

### Features

- Path tracing with NEE, stochastic progressive photon mapping, and bidirectional path tracing with multiple importance sampling (MIS) 
- GGX microfacet model for rough conductors and dielectrics
- Lambertian diffuse, perfect and rough specular reflection/transmission
- Point lights, area lights, directional/infinite lights, and importance sampled environment maps

### Possible future features
- [ ] Better light transport, e.g. vertex connection and merging (VCM) or path guiding
- [ ] Spectral rendering
- [ ] Volumetric rendering of homogeneous and non-homogeneous media
- [ ] More complex materials to simulate thin-film interference and subsurface scattering 
- [ ] Realistic camera models with depth of field
- [X] Improved 3D file format support

<img src="https://github.com/user-attachments/assets/7a506f16-6eb7-47a3-915d-0e50e10a10a8" width="1000">
<img src="https://github.com/user-attachments/assets/f5ee930d-79bd-454e-93cb-f40e30e87df1" width="1000">
<img src="https://github.com/user-attachments/assets/babe7799-2c19-42f3-a8b0-6aee2b81aa26" width="1000">
<img src="https://github.com/user-attachments/assets/e85a327f-d8a0-405a-9f87-e89e9cf13eb0" width="1000">
<img src="https://github.com/user-attachments/assets/550d16ac-9cd9-40b9-9de9-d058e334eac3" width="1000">
<img src="https://github.com/user-attachments/assets/e8e3438e-7544-4b8a-bff7-c10fd90632a8" width="1000">
