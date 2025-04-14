import numpy as np
import pickle
import gymnasium as gym
import os
from typing import Optional
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl


def split_data_by_mouse_id_and_trial(max_syllables: Optional[int] = None):
    script_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        script_path, "..", "data", "dopamine_level", "data_by_mouse_id.npy"
    )
    seqs, DAs, z_DAs, timestamps = np.load(data_path, allow_pickle=True)
    print("\n" + "=" * 50)
    print("Dataset Metadata".center(50))
    print("=" * 50)
    print(f"Mouse IDs: {list(seqs.keys())}")
    print(f"Dopamine levels available for: {list(DAs.keys())}")
    print(f"Z-scored dopamine levels available for: {list(z_DAs.keys())}")
    print(f"Timestamps available for: {list(timestamps.keys())}")
    print("=" * 50 + "\n")

    # store the data by mouse id
    for mouse_id in seqs.keys():
        print("\n" + "-" * 50)
        print(f"Processing data for mouse {mouse_id}".center(50))
        print("-" * 50)

        # Create a directory for each mouse if it doesn't exist
        mouse_dir = os.path.join(
            script_path, "..", "data", "dopamine_level", f"mouse_{mouse_id}"
        )
        os.makedirs(mouse_dir, exist_ok=True)

        # Extract data for this mouse
        mouse_data = {
            "sequences": seqs[mouse_id],
            "dopamine_levels": DAs[mouse_id],
            "z_scored_dopamine_levels": z_DAs[mouse_id],
            "timestamps": timestamps[mouse_id],
        }

        # Save data in pickle format with highest protocol for efficiency
        pickle_path = os.path.join(mouse_dir, "data.pickle")
        with open(pickle_path, "wb") as f:
            pickle.dump(mouse_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save data in numpy compressed format for numerical data
        numpy_path = os.path.join(mouse_dir, "data.npz")
        np.savez_compressed(
            numpy_path,
            sequences=seqs[mouse_id],
            dopamine_levels=DAs[mouse_id],
            z_scored_dopamine_levels=z_DAs[mouse_id],
            timestamps=timestamps[mouse_id],
        )

        # Get the max number of syllables in the data
        # if max_syllables is not provided, use the number of unique syllables in the data
        if max_syllables is None:
            max_syllables = len(np.unique(seqs[mouse_id]))
        else:
            max_syllables = max_syllables

        print(f"Max syllables: {max_syllables}")

        # Process the data and save it into a replay buffer for each mouse
        mouse_id_rb = ReplayBuffer(
            buffer_size=seqs[mouse_id].shape[0] * seqs[mouse_id].shape[1],
            observation_space=gym.spaces.Discrete(max_syllables),
            action_space=gym.spaces.Discrete(max_syllables),
        )

        # We will process the data by trial first,
        # add each valid transition pair to the replay buffer in each trial,
        # then save the replay buffer for each trial
        # then add the data from all trials to the replay buffer for the mouse
        # then save the replay buffer for the mouse

        # additionally, split the data by trial
        for trial_id in range(len(seqs[mouse_id])):
            print("\n" + "-" * 30)
            print(f"Processing trial {trial_id}".center(30))
            print("-" * 30)

            # Create a directory for each mouse if it doesn't exist
            mouse_trial_dir = os.path.join(
                script_path,
                "..",
                "data",
                "dopamine_level",
                f"mouse_{mouse_id}",
                f"trial_{trial_id}",
            )
            os.makedirs(mouse_trial_dir, exist_ok=True)

            # Extract data for this mouse and trial
            mouse_data = {
                "sequences": seqs[mouse_id][trial_id],
                "dopamine_levels": DAs[mouse_id][trial_id],
                "z_scored_dopamine_levels": z_DAs[mouse_id][trial_id],
                "timestamps": timestamps[mouse_id][trial_id],
            }

            # Save data in pickle format with highest protocol for efficiency
            pickle_path = os.path.join(mouse_trial_dir, "data.pickle")
            with open(pickle_path, "wb") as f:
                pickle.dump(mouse_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save data in numpy compressed format for numerical data
            numpy_path = os.path.join(mouse_trial_dir, "data.npz")
            np.savez_compressed(
                numpy_path,
                sequences=seqs[mouse_id][trial_id],
                dopamine_levels=DAs[mouse_id][trial_id],
                z_scored_dopamine_levels=z_DAs[mouse_id][trial_id],
                timestamps=timestamps[mouse_id][trial_id],
            )

            # Process the data and save it into a replay buffer for each trial
            trial_id_rb = ReplayBuffer(
                buffer_size=seqs[mouse_id][trial_id].shape[0],
                observation_space=gym.spaces.Discrete(max_syllables),
                action_space=gym.spaces.Discrete(max_syllables),
            )

            # Add each valid transition pair to the replay buffer
            for i in range(seqs[mouse_id][trial_id].shape[0] - 1):
                # only save the transition if the current state and next state are all within the max_syllables
                if (
                    seqs[mouse_id][trial_id][i] < max_syllables
                    and seqs[mouse_id][trial_id][i + 1] < max_syllables
                ):
                    trial_id_rb.add(
                        obs=seqs[mouse_id][trial_id][i],
                        next_obs=seqs[mouse_id][trial_id][i + 1],
                        action=seqs[mouse_id][trial_id][i + 1],
                        reward=DAs[mouse_id][trial_id][i + 1],
                        done=np.array([False]),
                        infos=[{}],
                    )

                    mouse_id_rb.add(
                        obs=seqs[mouse_id][trial_id][i],
                        next_obs=seqs[mouse_id][trial_id][i + 1],
                        action=seqs[mouse_id][trial_id][i + 1],
                        reward=DAs[mouse_id][trial_id][i + 1],
                        done=np.array([False]),
                        infos=[{}],
                    )

            # Save the replay buffer for each trial
            trial_id_rb_path = os.path.join(mouse_trial_dir, "trial_rb.pkl")
            save_to_pkl(trial_id_rb_path, trial_id_rb)

            print(f"✓ Data saved to {mouse_trial_dir}")

        # Save the replay buffer for the mouse
        mouse_id_rb_path = os.path.join(mouse_dir, "mouse_rb.pkl")
        save_to_pkl(mouse_id_rb_path, mouse_id_rb)

        print("\nReplay Buffer Statistics:")
        print(f"Observations shape: {mouse_id_rb.observations.shape}")
        print(f"Next observations shape: {mouse_id_rb.next_observations.shape}")
        print(f"Actions shape: {mouse_id_rb.actions.shape}")
        print(f"Rewards shape: {mouse_id_rb.rewards.shape}")
        print(f"Dones shape: {mouse_id_rb.dones.shape}")

        print(f"\n✓ All data for mouse {mouse_id} saved to {mouse_dir}")


if __name__ == "__main__":
    split_data_by_mouse_id_and_trial(max_syllables=10)
