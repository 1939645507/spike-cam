This directory stores third-party or separately developed encoder code.

- `Autoencoders-in-Spike-Sorting/`: bundled external AE repository
- `VAE/`: additional external model assets kept from the original project

The main experiment platform does not require these directories to run in
minimal mode. If optional dependencies are unavailable, the AE frontend falls
back to the built-in lightweight numpy implementation.
