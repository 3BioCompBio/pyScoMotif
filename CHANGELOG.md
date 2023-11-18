Changelog
---------

18/11/2023
- Added support for MMCIF files.

25/09/2023
- Replaced concurrent.futures.ProcessPoolExecutor parallelisation with joblib (there is a known deadlock issue with ProcessPoolExecutor that occurs frequently when many thousands of futures are submitted.)