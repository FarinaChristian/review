# How to run the algorithms
In the directories named after the papers, you can find the implementations of the algorithms.
To run the code, move the .py files from those directories into the same directory as the folders named constants and decoder.

## Papers

| Device        | Year          | DOI          | Implemented |
| ------------- |:-------------:| -------------|-------------| 
| (DSP/FPGA) board                                         | 2019 |10.1038/s41928-019-0258-6                 |Yes|
| AWR1642                                                  | 2020 |10.3390/s20102999                         |Yes (Hongjia)|
| TI IWR6843 ISK                                           | 2020 |10.1109/ICRA40945.2020.9197437            |No (deep learning)|
| TI board                                                 | 2021 |https://doi.org/10.1145/3478090           |Yes (only static)|
| TI AWR1642Boost                                          | 2021 |https://doi.org/10.1145/3478127           |No (deep learning)|
| TI AWR1843                                               | 2022 |10.1109/TMC.2022.3214721                  |No (deep learning)|
|e MoVi-Fi as a motion-robust vital signs monitoring system| 2022 |https://doi.org/10.1145/3447993.3483251   |No (deep learning)|
| TI IWR1843BOOST                                          | 2024 |https://doi.org/10.1145/3699766           |No (referencies) |
| IWR1443BOOST                                             | 2024 |https://doi.org/10.1038/s41598-024-77683-1|Yes|
| TI IWR6843 ISK                                           | 2025 |10.1109/RCAR65431.2025.11139413           |Yes|
| IWR1642BOOST                                             | 2025 |https://doi.org/10.1038/s41598-025-09112-w|Yes|

## Results

The table below summarizes results obtained by the methodologies on the full `mmStillDataset`. 
When a paper does not report BR or HR as part of its original output, the corresponding cell is marked `N/A`.

<table>
  <thead>
    <tr>
      <th rowspan="2">Subjects</th>
      <th rowspan="2"># Exp</th>
      <th colspan="2">Mercuri_2019_vital</th>
      <th colspan="2">wang_2020_remote</th>
      <th colspan="2">chen_2024_high</th>
      <th colspan="2">hao_2025_detection</th>
      <th colspan="2">FMWC</th>
    </tr>
    <tr>
      <th><em>BR MAE</em></th>
      <th><em>HR MAE</em></th>
      <th><em>BR MAE</em></th>
      <th><em>HR MAE</em></th>
      <th><em>BR MAE</em></th>
      <th><em>HR MAE</em></th>
      <th><em>BR MAE</em></th>
      <th><em>HR MAE</em></th>
      <th><em>BR MAE</em></th>
      <th><em>HR MAE</em></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>52</td>
      <td>0.55</td>
      <td>30.19</td>
      <td>0.40</td>
      <td>24,88</td>
      <td>N/A</td>
      <td>-</td>
      <td>0.82</td>
      <td>9.18</td>
      <td>0.45</td>
      <td>28.16</td>
    </tr>
    <tr>
      <td>2</td>
      <td>15</td>
      <td>0.69</td>
      <td>11.33</td>
      <td>0.86</td>
      <td>17.49</td>
      <td>N/A</td>
      <td>-</td>
      <td>0.96</td>
      <td>12.79</td>
      <td>0.92</td>
      <td>19.85</td>
    </tr>
    <tr>
      <td>3</td>
      <td>15</td>
      <td>0.51</td>
      <td>13.81</td>
      <td>0.64</td>
      <td>16.20</td>
      <td>N/A</td>
      <td>-</td>
      <td>0.94</td>
      <td>12.64</td>
      <td>0.68</td>
      <td>15.47</td>
    </tr>
    <tr>
      <td>4</td>
      <td>15</td>
      <td>0.99</td>
      <td>14.36</td>
      <td>0.85</td>
      <td>16.43</td>
      <td>N/A</td>
      <td>-</td>
      <td>1.05</td>
      <td>10.12</td>
      <td>0.91</td>
      <td>16.74</td>
    </tr>
    <tr>
      <td><strong>Average</strong></td>
      <td><strong>97</strong></td>
      <td><strong>0.68</strong></td>
      <td><strong>17.42</strong></td>
      <td><strong>0.69</strong></td>
      <td><strong>18.75</strong></td>
      <td><strong>N/A</strong></td>
      <td><strong>-</strong></td>
      <td><strong>0.94</strong></td>
      <td><strong>11.18</strong></td>
      <td><strong>0.74</strong></td>
      <td><strong>20.06</strong></td>
    </tr>
  </tbody>
</table>

## Surveys

| Year | DOI                                             |
| -----|-------------------------------------------------| 
| 2023 |10.1109/COMST.2023.3298300                       |
| 2023 |https://doi.org/10.1145/3627161                  |
| 2024 |10.1109/COMST.2024.3409556                       |
| 2021 |10.1109/JSEN.2021.3057450                        |
| 2021 |10.1109/ISAPE54070.2021.9753424                  |
| 2025 |10.3390/s25030602                                |
| 2025 |https://doi.org/10.3390/s25123706                |
| 2025 |https://doi.org/10.1016/j.compeleceng.2025.110696|
