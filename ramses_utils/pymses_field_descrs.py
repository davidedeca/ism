from pymses.sources.ramses import output
self.amr_field_descrs_by_file = \
    {"3D": {
        "hydro" : [ output.Scalar("rho", 0),
                    output.Vector("vel", [1, 2, 3]),
                    output.Scalar("P", 4),
                    output.Scalar("ion1", 5),
                    output.Scalar("ion2", 6),
                    output.Scalar("ion3", 7),
                    output.Scalar("ion4", 8),
                    output.Scalar("ion5", 9),
                    output.Scalar("ion6", 10),
                    output.Scalar("ion7", 11),
                    output.Scalar("ion8", 12),
                    output.Scalar("ion9", 13)],
        "rt"    : [ output.Scalar("rad_density1", 0), output.Vector("rad_flux1", [1, 2, 3]),
                    output.Scalar("rad_density2", 4), output.Vector("rad_flux2", [5, 6, 7]),
                    output.Scalar("rad_density3", 8), output.Vector("rad_flux3", [9, 10, 11]),
                    output.Scalar("rad_density4", 12), output.Vector("rad_flux4", [13, 14, 15]),
                    output.Scalar("rad_density5", 16), output.Vector("rad_flux5", [17, 18, 19]),
                    output.Scalar("rad_density6", 20), output.Vector("rad_flux6", [21, 22, 23]),
                    output.Scalar("rad_density7", 24), output.Vector("rad_flux7", [25, 26, 27]),
                    output.Scalar("rad_density8", 28), output.Vector("rad_flux8", [29, 30, 31]),
                    output.Scalar("rad_density9", 32), output.Vector("rad_flux9", [33, 34, 35]),
                    output.Scalar("rad_density10", 36), output.Vector("rad_flux10", [37, 38, 39])]
          }
    }