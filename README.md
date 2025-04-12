# mte546-project

By: Henchel Santillan, Jason Xue, Joshua Alexander Auw-Yang

## Setup

1) First create a python virtual environment. Make sure you have python 3.10 or above. Run the following command on *WINDOWS OS* to create an environment named "env" for a python environment running on python 3.12:

    ```py -3.12 -m venv "env"```

2) Now activate/enter the environment (for *WINDOWS OS*).

    ```env/Scripts/Activate```

3) Now download the required python libraries from the `requirements.txt`:

    ```pip install -r requirements.txt```

4) Go into the following library file and replace the entire `predict_update()` method with the code below (there is an error with the imported EKF library's implementation):

    * path: `env\Lib\site-packages\filterpy\kalman\EKF.py`


        ```Python
            def predict_update(self, z, HJacobian, Hx, args=(), hx_args=(), u=0):
                """ Performs the predict/update innovation of the extended Kalman
                filter.

                Parameters
                ----------

                z : np.array
                    measurement for this step.
                    If `None`, only predict step is perfomed.

                HJacobian : function
                    function which computes the Jacobian of the H matrix (measurement
                    function). Takes state variable (self.x) as input, along with the
                    optional arguments in args, and returns H.

                Hx : function
                    function which takes as input the state variable (self.x) along
                    with the optional arguments in hx_args, and returns the measurement
                    that would correspond to that state.

                args : tuple, optional, default (,)
                    arguments to be passed into HJacobian after the required state
                    variable.

                hx_args : tuple, optional, default (,)
                    arguments to be passed into Hx after the required state
                    variable.

                u : np.array or scalar
                    optional control vector input to the filter.
                """
                #pylint: disable=too-many-locals

                if not isinstance(args, tuple):
                    args = (args,)

                if not isinstance(hx_args, tuple):
                    hx_args = (hx_args,)

                if np.isscalar(z) and self.dim_z == 1:
                    z = np.asarray([z], float)
                F = self.F
                B = self.B
                P = self.P
                Q = self.Q
                R = self.R
                x = self.x

                H = HJacobian(x, *args)

                # predict step
                x = dot(F, x) + dot(B, u)
                P = dot(F, P).dot(F.T) + Q

                # save prior
                self.x_prior = np.copy(self.x)
                self.P_prior = np.copy(self.P)

                # update step
                PHT = dot(P, H.T)

                self.S = dot(H, PHT) + R
                self.SI = linalg.inv(self.S)
                self.K = dot(PHT, self.SI)

                self.y = z - Hx(x, *hx_args)


                self.x = x.reshape(-1,1) + dot(self.K, self.y)

                I_KH = self._I - dot(self.K, H)
                self.P = dot(I_KH, P).dot(I_KH.T) + dot(self.K, R).dot(self.K.T)

                # save measurement and posterior state
                self.z = deepcopy(z)
                self.x_post = self.x.copy()
                self.P_post = self.P.copy()

                # set to None to force recompute
                self._log_likelihood = None
                self._likelihood = None
                self._mahalanobis = None
        ```

5) Download the sleeping dataset .zip file [here](https://drive.google.com/file/d/16mlB-vbB7vNToeu3C7gVl0XBw33yO7VD/view?usp=sharing) and unzip it into your `Downloads/` folder. Make sure it is unzipped such that the directory path follows a format like the following:

    `Downloads/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0/heart_rate/...`


## Running the code

The main file in this repository is `sleep.py`. After entering the environment, run the following in the terminal:

```python sleep.py``` 