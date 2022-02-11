# Gate sequences to approximate rz gates with arguments ^n.
# Key is n so 0->rz(pi/1), 1->rz(pi/2), 2->rx(pi/4) ...
# Note that rz(theta) is a rotation by theta/2 about the Z axis in our convention.
get_pi_over_2_to_the_n_rz_gate = [
    "SS",
    "S",
    "SSXHSTHSTHTHSTHTHSTHSTHTHTHSTHSTHSTHTHTHSTHTHTHSTHSTHSTHSTHSTHTHSTHTHSTHSTHTHSTHTHSTHSTHSTHTHSTHSTHSTHSTHSTHTHTHTHTHTHSTHTHSTHTHTHSTHTHTHTHSTHTHTHSTHSTHTHSTHSTHTHSTHSTHSTHTHTHSTHTHTHTHTHTHTHTHSTHTHSTHTHSTHSTHSTHTHSTHTHTHSTHTHSTHTHTHSTHTHTHSTHTHSTHTHSTHSTHTHTHSTHSTHSTHTHTHTHTHTHSTHSTHTHSTHTHSTHSTHSTHTHTHSTHTHSTHSTHSTHTHTHTHS",  # noqa: E501
    "HTHSTHTHTHSTHTHSTHSTHSTHSTHTHSTHSTHTHSTHSTHTHTHTHSTHSTHTHSTHTHTHSTHSTHSTHSTHTHTHSTHTHSTHTHTHTHTHSTHSTHTHSTHSTHSTHTHTHTHSTHSTHTHSTHSTHTHTHTHSTHSTHSTHSTHSTHSTHSTHSTHTHTHSTHSTHTHTHTHTHTHSTHSTHSTHTHTHSTHSTHSTHTHSTHSTHSTHSTHTHSTHSTHSTHTHTHTHTHSTHSTHTHSTHTH",  # noqa: E501
    "XHSTHSTHSTHSTHSTHSTHSTHTHTHTHTHSTHSTHSTHTHSTHSTHTHSTHTHTHTHSTHTHTHSTHSTHTHTHSTHTHTHTHSTHSTHTHTHTHTHSTHSTHTHTHTHSTHTHTHTHTHSTHSTHTHSTHTHTHTHTHSTHTHTHSTHTHSTHSTHTHSTHTHTHSTHTHSTHSTHTHSTHSTHSTHSTHSTHSTHTHTHTHSTHSTHTHTHTHSTHSTHSTHTHTHTHSTHSTHTHTHSTHTHTH",  # noqa: E501
    "HTHSTHTHTHSTHSTHTHTHSTHTHTHTHSTHSTHTHTHSTHSTHSTHSTHSTHSTHSTHTHTHSTHTHSTHTHSTHSTHSTHSTHTHTHSTHTHTHTHSTHSTHSTHSTHTHSTHSTHSTHTHSTHSTHTHSTHSTHTHSTHTHSTHTHTHSTHTHSTHSTHSTHTHSTHTHTHTHSTHSTHTHSTHTHTHTHTHTHTHSTHTHTHSTHTHSTHSTHTHTHSTHSTHTHTHSTHSTHTHSTHTHTHSTHSTH",  # noqa: E501
    "SSSHSTHTHTHSTHTHTHTHSTHTHSTHSTHSTHSTHTHTHSTHSTHTHSTHSTHSTHSTHTHSTHTHSTHTHTHSTHSTHTHTHTHTHSTHSTHSTHSTHTHSTHSTHTHSTHSTHSTHTHSTHTHTHSTHSTHSTHSTHTHTHTHTHSTHTHSTHTHSTHTHSTHSTHTHSTHSTHTHTHSTHSTHSTHTHTHSTHSTHSTHSTHSTHTHTHSTHSTHTHTHTHTHTHSTHTHTHTHSTHSTHSTHSTHSTHS",  # noqa: E501
    "SSSHSTHSTHSTHTHTHTHSTHTHSTHSTHSTHTHTHSTHSTHTHTHSTHTHSTHTHTHTHTHTHTHSTHSTHSTHTHSTHTHSTHTHTHTHSTHTHTHSTHTHSTHTHTHTHSTHSTHSTHTHTHSTHSTHSTHSTHTHSTHSTHSTHSTHTHSTHTHSTHSTHTHTHTHTHSTHTHTHTHSTHSTHSTHTHSTHSTHTHTHSTHTHTHTHTHSTHSTHTHTHTHTHSTHTHTHTHSTHTHTHTHTHTHTHS",  # noqa: E501
    "SSSHTHSTHSTHSTHSTHSTHTHSTHTHTHSTHTHSTHTHTHSTHSTHSTHSTHTHSTHTHTHSTHTHSTHTHSTHTHSTHSTHTHTHTHTHTHTHSTHTHSTHTHSTHSTHSTHSTHTHSTHSTHTHSTHSTHTHSTHTHSTHTHSTHTHTHSTHSTHTHSTHSTHSTHTHTHTHTHTHTHTHTHTHSTHTHSTHSTHTHSTHSTHTHSTHTHSTHSTHTHTHTHTHTHTHTHSTHSTHTHTHTHSTHSTHTHTHS",  # noqa: E501
    "HTHSTHSTHTHSTHTHTHSTHTHSTHTHTHSTHTHSTHTHTHTHTHSTHTHSTHTHTHSTHSTHSTHSTHSTHTHSTHTHSTHSTHSTHTHTHTHTHTHTHSTHSTHSTHTHTHTHTHTHSTHTHSTHSTHTHTHTHTHSTHSTHTHSTHTHSTHSTHTHSTHSTHTHSTHSTHSTHSTHSTHSTHTHTHTHTHSTHTHTHSTHTHSTHSTHSTHTHTHTHSTHSTHTHTHTHSTHTHSTHST",  # noqa: E501
    "XHSTHSTHSTHSTHSTHSTHTHSTHSTHSTHSTHSTHTHSTHTHSTHTHSTHSTHSTHSTHSTHTHTHTHTHSTHSTHTHSTHSTHTHTHTHTHTHTHSTHTHSTHSTHTHTHTHTHSTHTHSTHTHTHSTHTHSTHSTHSTHTHSTHTHTHTHSTHTHTHSTHSTHSTHTHTHTHTHTHTHTHSTHTHTHTHSTHSTHSTHTHTHSTHSTHSTHSTHTHTHTHSTHSTHTHSTHSTHTHTHTHSTHTHSTH",  # noqa: E501
    "STHTHTHTHTHTHSTHTHSTHTHSTHTHTHSTHSTHTHTHSTHTHTHTHSTHTHSTHSTHSTHTHSTHTHTHSTHSTHTHSTHTHSTHTHSTHTHTHSTHSTHSTHSTHSTHSTHTHSTHSTHTHTHSTHSTHSTHSTHTHTHTHTHTHTHSTHSTHTHTHTHTHSTHSTHSTHSTHSTHTHSTHTHTHTHSTHSTHSTHTHSTHTHTHSTHTHSTHSTHSTHTHTHTHSTHTHSTHTHTHTHTHSTH",  # noqa: E501
    "STHSTHSTHTHTHTHTHSTHTHTHTHSTHTHTHSTHSTHTHTHTHSTHTHTHTHTHTHSTHTHTHTHTHSTHSTHTHSTHSTHTHSTHSTHSTHSTHTHSTHTHSTHSTHTHTHSTHTHTHTHTHSTHTHTHSTHSTHSTHTHSTHSTHTHTHSTHSTHSTHTHSTHSTHTHSTHSTHSTHSTHTHSTHTHSTHSTHTHTHTHTHTHTHTHTHSTHSTHSTHSTHSTHTHTHSTHSTHTHSTHSTHSTH",  # noqa: E501
    "SXHTHSTHSTHTHTHSTHSTHTHTHSTHTHSTHTHTHTHTHSTHTHSTHSTHTHTHSTHTHTHSTHSTHTHSTHTHTHSTHTHTHSTHTHSTHSTHTHTHSTHSTHSTHSTHTHSTHTHSTHTHTHSTHTHSTHTHSTHSTHSTHSTHTHTHSTHTHSTHSTHSTHSTHSTHSTHTHSTHTHSTHTHTHSTHTHSTHSTHSTHSTHTHSTHTHTHSTHTHTHSTHSTHSTHTHTHTHSTHTHTHSTHSTHSTHSTH",  # noqa: E501
    "SSHSTHSTHSTHTHTHTHSTHSTHSTHTHSTHSTHTHSTHSTHTHSTHTHTHTHSTHTHTHTHTHSTHTHTHSTHTHTHTHTHSTHSTHTHTHTHSTHSTHTHTHSTHTHSTHTHTHTHSTHTHTHSTHSTHTHTHTHTHSTHSTHTHTHSTHSTHTHTHSTHTHSTHSTHSTHSTHTHTHSTHSTHSTHTHTHSTHSTHTHTHTHTHSTHSTHSTHTHSTHSTHTHTHTHTHTHSTHTHSTHS",  # noqa: E501
    "HSTHSTHSTHSTHTHSTHSTHSTHSTHTHTHTHSTHTHTHTHTHTHTHSTHTHSTHTHSTHSTHTHTHSTHTHTHSTHTHSTHTHTHSTHSTHTHSTHTHSTHTHTHTHSTHTHTHTHSTHTHTHSTHSTHSTHSTHTHTHSTHTHSTHTHSTHSTHTHTHTHTHTHSTHTHSTHSTHTHTHTHTHSTHSTHSTHSTHTHSTHTHTHTHSTHSTHTHTHTHSTHTHTHTHTHTHTHTHTHSTH",  # noqa: E501
    "SSSXHTHSTHSTHTHSTHTHSTHTHSTHSTHTHSTHSTHTHTHSTHSTHSTHSTHTHTHTHSTHTHSTHTHTHTHTHTHTHTHTHSTHSTHTHTHSTHTHSTHTHTHSTHSTHSTHTHTHTHTHSTHSTHTHSTHSTHSTHTHTHTHSTHSTHSTHTHTHTHSTHTHTHSTHSTHTHTHTHSTHTHTHTHTHTHSTHSTHTHSTHTHSTHSTHSTHTHSTHSTHTHTHSTHSTHTHTHTHSTHTHSTHTHTHTH",  # noqa: E501
    "SHTHSTHTHSTHTHSTHTHTHSTHSTHTHSTHTHTHTHSTHTHSTHTHSTHTHTHSTHSTHSTHSTHTHTHSTHTHTHSTHTHTHSTHSTHSTHTHTHTHTHTHSTHSTHTHSTHTHSTHSTHSTHTHSTHSTHSTHTHTHTHTHSTHTHTHTHTHSTHSTHTHTHSTHSTHTHSTHTHTHSTHTHTHSTHSTHTHTHTHSTHTHTHTHTHSTHSTHTHSTHSTHTHSTHSTHSTHSTHTHTHS",  # noqa: E501
    "SSSTHSTHTHSTHTHSTHSTHSTHTHTHSTHSTHSTHSTHTHTHSTHSTHTHTHTHTHTHSTHTHSTHSTHTHSTHTHSTHTHSTHTHTHSTHSTHTHTHSTHTHSTHSTHTHSTHSTHSTHSTHSTHSTHTHSTHSTHTHSTHTHSTHSTHSTHSTHTHSTHTHTHSTHTHTHSTHSTHSTHTHSTHTHSTHSTHSTHSTHSTHTHSTHTHTHSTHSTHSTHSTHTHTHSTHTHTHSTHSTHTHTHSTHSTHSTHSTHSTH",  # noqa: E501
    "SSSHSTHSTHTHSTHSTHSTHTHSTHTHTHTHSTHSTHSTHTHSTHTHTHTHSTHSTHTHSTHSTHSTHTHTHTHTHTHSTHSTHSTHSTHSTHTHSTHSTHSTHSTHSTHSTHSTHTHTHSTHTHTHTHTHTHTHTHSTHTHSTHTHSTHSTHSTHSTHSTHTHTHTHSTHTHSTHSTHTHTHTHSTHSTHTHTHSTHSTHSTHSTHTHSTHSTHSTHTHTHTHSTHTHSTHSTHSTHTHTHTHSTHTHTHSTHSTHS",  # noqa: E501
    "SSSXHTHTHTHTHTHTHSTHTHTHSTHTHSTHTHTHTHSTHSTHSTHTHTHTHSTHTHSTHTHTHTHSTHSTHTHTHTHSTHSTHTHTHSTHSTHSTHTHTHTHTHSTHSTHSTHSTHSTHTHSTHTHSTHTHTHSTHTHSTHSTHSTHSTHTHTHTHTHTHTHSTHSTHTHSTHTHSTHTHSTHTHTHSTHTHTHSTHSTHSTHSTHSTHTHTHTHSTHTHTHSTHTHTHSTHS",  # noqa: E501
    "SSSHTHSTHSTHSTHSTHTHSTHTHSTHSTHSTHTHTHTHSTHTHTHSTHTHTHSTHSTHSTHTHTHTHSTHSTHSTHSTHSTHTHTHSTHSTHTHTHTHSTHTHTHSTHSTHSTHSTHTHTHTHTHSTHSTHTHSTHTHTHSTHSTHTHSTHSTHTHSTHSTHSTHSTHSTHSTHTHSTHSTHSTHSTHTHSTHTHSTHTHSTHTHSTHTHSTHTHTHSTHSTHSTHSTHSTHTHTHSTHTHTHSTHSTHSTHSTH",  # noqa: E501
    "SHSTHSTHSTHSTHSTHTHTHSTHSTHSTHTHTHSTHSTHTHTHTHTHTHSTHSTHSTHTHSTHTHSTHTHSTHSTHSTHSTHTHSTHSTHTHSTHSTHSTHSTHTHSTHSTHTHTHSTHTHTHSTHTHTHTHTHTHTHTHTHSTHSTHTHTHTHSTHTHSTHTHTHSTHTHTHSTHTHSTHTHTHSTHSTHSTHSTHSTHSTHSTHSTHSTHTHSTHTHSTHTHSTHTHSTHTHTHTHSTHSTHTHSTHTHTHSTHSTHTHTH",  # noqa: E501
    "XTHTHTHTHSTHSTHTHTHTHTHTHTHSTHTHSTHTHTHTHSTHSTHTHSTHSTHSTHTHSTHTHTHSTHSTHSTHTHSTHSTHSTHSTHTHSTHTHTHTHTHTHTHSTHTHTHSTHSTHTHTHSTHSTHSTHSTHSTHTHSTHTHSTHTHTHSTHTHSTHTHSTHSTHTHSTHTHTHTHSTHSTHTHTHSTHSTHSTHSTHSTHSTHTHSTHSTHSTHTHTHSTHSTHTHTHSTHSTHSTHSTHSTHSTHTHTHSTHTHTHTHSTHSTHSTH",  # noqa: E501
    "SXTHSTHSTHSTHTHSTHTHSTHTHSTHSTHTHSTHTHSTHTHSTHTHSTHSTHTHSTHSTHSTHTHTHTHSTHSTHSTHSTHSTHTHSTHTHSTHSTHTHSTHSTHSTHSTHSTHSTHSTHTHTHTHSTHTHTHTHTHSTHTHTHTHTHTHSTHTHSTHTHTHSTHSTHTHTHTHSTHTHTHSTHSTHSTHSTHSTHTHTHTHSTHSTHTHTHSTHSTHTHSTHSTHTHSTHSTHSTHTHSTHSTHTHSTHSTHSTHSTHSTHTHTHSTHSTH",  # noqa: E501
    "SHSTHSTHSTHTHSTHSTHTHTHSTHSTHSTHSTHTHTHTHTHSTHSTHSTHSTHTHTHSTHSTHSTHTHSTHSTHTHSTHTHTHTHTHSTHSTHSTHTHTHSTHTHSTHSTHSTHTHSTHTHSTHTHTHSTHSTHSTHTHTHTHSTHTHSTHSTHTHTHSTHSTHSTHTHTHSTHTHSTHTHSTHTHTHSTHSTHTHTHSTHTHTHSTHSTHSTHSTHSTHSTHTHSTHTHSTHSTHTHTHTHTHTHTHSTHSTHSTHSTHTHSTHTHTHSTHSTHSTHTHSTHTH",  # noqa: E501
    "XHTHSTHSTHTHSTHTHSTHTHTHTHSTHTHTHSTHTHTHSTHSTHSTHTHSTHSTHSTHSTHTHSTHSTHTHTHTHSTHTHTHSTHTHTHSTHTHTHTHTHTHTHSTHTHTHSTHSTHSTHSTHSTHSTHSTHSTHTHTHSTHTHTHSTHSTHTHTHSTHTHTHSTHTHSTHTHSTHSTHSTHTHSTHSTHTHTHTHSTHTHSTHTHSTHTHSTHSTHTHSTHSTHSTHTHTHTHTHSTHTHSTHSTHSTHTHSTHSTHSTHSTHTHTHSTHTHSTHSTHTHSTHST",  # noqa: E501
    "SSXHTHSTHTHSTHTHTHSTHSTHSTHSTHTHSTHSTHSTHSTHTHTHSTHSTHSTHTHSTHSTHTHTHTHSTHTHTHTHSTHSTHSTHTHSTHTHTHTHTHTHSTHTHTHSTHSTHSTHSTHSTHTHSTHTHTHSTHTHTHSTHSTHTHSTHTHSTHSTHTHTHSTHSTHTHSTHTHTHTHSTHTHSTHSTHTHSTHSTHSTHSTHSTHTHTHTHTHTHTHTHSTHTHSTHTHTHSTHTHSTHSTHTHTHSTHSTHTHTHTHTHSTHTHSTHSTHTHTHSTHSTHSTHS",  # noqa: E501
    "SXTHSTHTHSTHSTHSTHSTHTHSTHSTHTHSTHSTHTHTHSTHTHTHSTHTHTHTHTHSTHTHSTHSTHTHSTHTHTHSTHTHSTHSTHSTHTHTHTHTHSTHTHTHTHSTHSTHTHSTHTHSTHSTHSTHSTHTHSTHSTHSTHTHTHTHTHTHTHSTHSTHSTHTHSTHSTHSTHSTHSTHTHTHSTHSTHTHTHSTHSTHSTHTHSTHTHTHSTHSTHSTHTHSTHTHSTHSTHTHTHSTHSTHTHSTHSTHTHTHSTHSTHSTHSTHTHTHTHSTHTHTHSTHTHSTHSTHS",  # noqa: E501
    "SSXTHSTHSTHSTHSTHSTHSTHSTHTHSTHSTHTHTHTHSTHTHTHTHTHTHTHTHSTHTHSTHSTHTHTHSTHTHTHTHTHTHSTHSTHSTHSTHTHSTHTHSTHSTHSTHTHSTHTHSTHTHSTHTHSTHTHTHTHTHTHTHSTHTHTHSTHTHSTHSTHTHSTHTHSTHTHTHSTHSTHTHSTHSTHTHTHSTHSTHTHSTHTHTHSTHTHSTHSTHTHTHTHSTHTHSTHSTHSTHTHSTHTHSTHSTHSTHTHTHTHSTHSTHSTHSTHTH",  # noqa: E501
]