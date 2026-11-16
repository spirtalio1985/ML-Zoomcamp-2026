# Data

## Source

**Online Shoppers Purchasing Intention Dataset**, UCI Machine Learning Repository (dataset id 468):
<https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset>

* Citation: Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset* [Dataset].
  UCI Machine Learning Repository. <https://doi.org/10.24432/C5F88Q>
* Paper that introduced the data: Sakar, C. O., Polat, S. O., Katircioglu, M., & Kastro, Y. (2019).
  Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM
  recurrent neural networks. *Neural Computing and Applications*, 31(10), 6893–6908.
  <https://doi.org/10.1007/s00521-018-3523-0>
* Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the UCI page.

## File in this folder

| File | Size | Rows | Columns | SHA-256 |
|---|---|---|---|---|
| `online_shoppers_intention.csv` | 1,072,063 bytes | 12,330 (plus header) | 18 | `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9` |

The file is committed to the repository, so nothing has to be downloaded to run the project.

**Provenance of this copy.** The file was taken from a public GitHub copy of the UCI file
(`sidneykung/online_shopper_intentions`, `data/online_shoppers_intention.csv`, commit `95fd58c`).
Before using it, it was checked that:

* a second, independent public copy (`sharmaroshan/Online-Shoppers-Purchasing-Intention`) is
  byte-for-byte identical, and a third (`semnan-university-ai/Online-Shoppers-Purchasing-Intention`)
  has identical content with Unix instead of Windows line endings;
* it matches the summary published on the UCI page: 12,330 sessions, 17 features plus the target,
  no missing values, 10,422 sessions without and 1,908 sessions with a purchase.

It was not compared byte-for-byte with a fresh download from the UCI website. To do that, download
`online_shoppers_intention.csv` from the UCI page above and compare the checksum (if only the line
endings differ, the content is still the same):

```bash
sha256sum data/online_shoppers_intention.csv        # Linux
shasum -a 256 data/online_shoppers_intention.csv    # macOS
certutil -hashfile data\online_shoppers_intention.csv SHA256   # Windows
```

`.gitattributes` tells git not to convert the line endings of this file, so the checksum stays the
same on every operating system.

## Columns

Each row is one browsing session; according to the dataset description every session belongs to a
different user, and the sessions cover a one-year period. The project renames the columns to
`snake_case` (see `COLUMN_RENAMES` in `features.py`).

| Column in the CSV | Name in the project | Type | Meaning |
|---|---|---|---|
| `Administrative` | `administrative` | count | Number of account-management pages visited |
| `Administrative_Duration` | `administrative_duration` | seconds | Time spent on account-management pages |
| `Informational` | `informational` | count | Number of pages about the shop itself (site, contact and address information) |
| `Informational_Duration` | `informational_duration` | seconds | Time spent on informational pages |
| `ProductRelated` | `product_related` | count | Number of product-related pages visited |
| `ProductRelated_Duration` | `product_related_duration` | seconds | Time spent on product-related pages |
| `BounceRates` | `bounce_rates` | 0–1 | Average Google Analytics bounce rate of the pages visited |
| `ExitRates` | `exit_rates` | 0–1 | Average Google Analytics exit rate of the pages visited |
| `PageValues` | `page_values` | ≥ 0 | Average Google Analytics page value of the pages visited |
| `SpecialDay` | `special_day` | 0–1 | Closeness of the visit to a special day (e.g. Valentine's Day, Mother's Day) |
| `Month` | `month` | category | Month of the visit (no sessions from January or April; June is spelled `June`) |
| `OperatingSystems` | `operating_systems` | code 1–8 | Operating system of the visitor |
| `Browser` | `browser` | code 1–13 | Browser of the visitor |
| `Region` | `region` | code 1–9 | Geographic region of the visitor |
| `TrafficType` | `traffic_type` | code 1–20 | Traffic source that brought the visitor to the site |
| `VisitorType` | `visitor_type` | category | `Returning_Visitor`, `New_Visitor` or `Other` |
| `Weekend` | `weekend` | TRUE/FALSE | Whether the visit happened on a weekend |
| `Revenue` | `revenue` | TRUE/FALSE | **Target**: whether the session ended with a purchase |

The UCI page does not document what each integer code stands for (which browser is "2", which
region is "3", …), so the codes are used as anonymous categories.
