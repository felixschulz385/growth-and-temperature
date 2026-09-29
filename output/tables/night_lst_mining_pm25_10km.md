|                                        | (1) TWFE   | (2) 2SLS   | (3) RF: Temp   | (4) TWFE + PM2.5   | (5) RF: PM2.5   |
|:---------------------------------------|:-----------|:-----------|:---------------|:-------------------|:----------------|
| log(1+NTL)                             | -0.0227**  | -0.1040    |                | -0.0221*           |                 |
|                                        | (0.0110)   | (0.1632)   |                | (0.0114)           |                 |
| Mines within 20 km                     |            |            | -0.0070        |                    | -0.2710**       |
|                                        |            |            | (0.0111)       |                    | (0.1365)        |
| PM2.5                                  |            |            |                | 0.0032*            |                 |
|                                        |            |            |                | (0.0017)           |                 |
| First stage: Mines 20 km -> log(1+NTL) |            | 0.0672***  |                |                    |                 |
|                                        |            | (0.0122)   |                |                    |                 |
| First-stage F (cluster-robust)         |            | 30.1       |                |                    |                 |
| Dep. var.                              | Night LST  | Night LST  | Night LST      | Night LST          | PM2.5           |
| Pixel FE                               | Yes        | Yes        | Yes            | Yes                | Yes             |
| Country x year FE                      | Yes        | Yes        | Yes            | Yes                | Yes             |
| Observations                           | 25,268,139 | 25,268,139 | 25,268,139     | 25,268,073         | 25,268,073      |
| Pixels                                 | 1,205,481  | 1,205,481  | 1,205,481      | 1,205,471          | 1,205,471       |
| Countries (clusters)                   | 253        | 253        | 253            | 253                | 253             |
| Years                                  | 2002-2022  | 2002-2022  | 2002-2022      | 2002-2022          | 2002-2022       |

SE clustered by country in parentheses. * p<0.1, ** p<0.05, *** p<0.01. 10 km grid, full panel. Cols (4)-(5) restricted to pixel-years with PM2.5.
Col (1) re-estimated on the PM2.5 sample: -0.0227** (0.0110). First stage on the PM2.5 sample: 0.0672 (0.0122), F = 30.1.
