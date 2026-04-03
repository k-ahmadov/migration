### Analysis of Coulomb Failure Function

In this notebook I import the 3DEC results for fluid pressure, normal and shear stresses along the fracture. From them, I compute the following Coulomb Failure Function (CFF):

$$ CFF(x,t) = \tau(x,t) - \mu( \sigma_n(x,t) - P(x,t) ). $$

The failure is reached when $CFF$ vanishes.

### Contribution of each variable on the CFF

I also estimate the contribution of each variable on the Coulomb stress change. The contribution of shear stress can be computed with the following equation: 
$$ SSC = \frac{\Delta \tau(x, t_i)}{-CFF(x, t_i)}, $$
where $SSC$ is the Shear Stress Contribution.

If $SCC$ equals to 1, the failure is driven by shear stress change.

Similarly, to compute the contribution of fluid pressure we can use the following:
$$ FPC = \frac{\Delta \P(x, t_i)}{-CFF(x, t_i)}. $$

And, the contribution of normal stess:
$$ NSC = \frac{\Delta \sigma_n(x, t_i)}{-CFF(x, t_i)}. $$
