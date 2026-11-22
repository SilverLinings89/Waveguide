#!/usr/bin/env python3
"""PML profile and its steps between quadrature points.

Dissertation: figure "Comparison of constant sigma and sigma(x) = 5 x^3" (section PML).
Produces: pml_profile.png

No simulation is involved. The figure illustrates why a larger sigma_max of an increasing PML profile leads to
reflections again: the profile is only evaluated at the quadrature points, and the jump between two neighboring
points grows towards the outer boundary.
"""
import numpy as np
import common


def main():
    arguments = common.argument_parser(__doc__).parse_args()
    common.prepare(arguments, "pml_profile")
    plt = common.pyplot()
    sigma_max, order, h = 5.0, 3, 0.25
    x = np.linspace(0, 1, 400)
    points = np.arange(0, 1 + 1e-12, h)
    profile = lambda t: sigma_max * t ** order

    figure, axis = plt.subplots(figsize=(8, 4.2))
    axis.plot(x, profile(x), color="tab:blue", label=r"$\sigma(x) = %g\,x^%d$" % (sigma_max, order))
    axis.plot(x, np.ones_like(x), color="tab:green", label=r"constant $\sigma = 1$")
    # Steps of the profile between neighboring evaluation points.
    for left, right in zip(points[:-1], points[1:]):
        axis.plot([left, right], [profile(left), profile(left)], color="tab:red", linestyle="--")
        axis.annotate("", xy=(right, profile(right)), xytext=(right, profile(left)),
                      arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=1.5))
    axis.annotate("", xy=(0, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=1.5))
    axis.set_xticks(points)
    axis.set_xlabel("depth in the PML (thickness 1), evaluation points every h = %g" % h)
    axis.set_ylabel(r"$\sigma$")
    axis.legend()
    common.save(figure, arguments, "pml_profile.png")


if __name__ == "__main__":
    main()
