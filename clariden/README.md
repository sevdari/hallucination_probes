## some config files


there are some config files related to the clariden cluster, e.g. [Dockerfile](./Dockerfile) of the image, should be some sbatch scripts etc.


The base `.sqsh` image with main branch cloned is saved at `/capstor/scratch/cscs/tkwiecinski/hallucination-probes/base.sqsh`. It might not have the latest repo version though. 




Notice, that there is no uv installed in the container (it seems that uv doesn't really like working with torch and global env).