# PROACT: PROjection and Activation Constrained Training for poisoning-resilient continual learning

BrainWash [1] is a recently proposed data poisoning attack designed to induce forgetting in task-incremental learners. This repository extends the [BrainWash codebase](https://github.com/mint-vu/Brainwash) with the implementation of PROACT, a defense strategy to counter the attack.

For each continual learning method, the defense logic is implemented in the `defend_step` function. The `ewc_pipeline.sh` script shows how to execute PROACT within the EWC pipeline.

## References

[1] Abbasi, A., Nooralinejad, P., Pirsiavash, H., & Kolouri, S. (2024). Brainwash: A poisoning attack to forget in continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 24057-24067).