# PATIENT COVERAGE AUDIT REPORT

*Generated: 2026-10-06 11:43 UTC*

---

## 1. Dataset Overview

| Metric | Value |
|---|---|
| Total patients in manifest | 245 |
| Patients with WSI (all) | 245 |
| Consensus-annotated patients | 113 |
| Unlabeled extension patients | 132 |
| Total annotation rows (all statuses) | 28,395 |
| Valid crops in metadata | 20,852 |
| Excluded artefact | 6,544 |
| Excluded no-consensus | 683 |
| Excluded spicule | 316 |
| Total crops on disk | 20,852 |
| Patients with crops on disk | 113 |
| Patients with zero crops | 132 |
| Consensus patients with zero crops | 0 |
| Orphan crop directories | 0 |
| Duplicate crop paths | 0 |
| Model checkpoint present | YES |
| Patients with model predictions | 0 |
| Patients with XAI artifacts | 0 |

---

## 2. Crop Status Distribution

- **ZERO_ANNOTATIONS**: 132 patients
- **AVAILABLE**: 113 patients

---

## 3. Crop Count Distribution (Consensus Cohort, N=113)

| Statistic | Value |
|---|---|
| count | 113.0 |
| mean | 184.5 |
| std | 110.5 |
| min | 2.0 |
| 25% | 94.0 |
| 50% | 180.0 |
| 75% | 259.0 |
| max | 593.0 |

---

## 4. Per-Patient Coverage Table

| patient_id_prefix | split | class | consensus | ann_rows | crops_meta | crops_disk | status | pred | xai |
|---|---|---|---|---|---|---|---|---|---|
| `002BD524C907A2A5` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `006F52E9102A8D3B` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `02522A2B2726FB0A` | train | AML | Y | 196 | 104 | 104 | AVAILABLE | N | N |
| `02E74F10E0327AD8` | train | CML | Y | 276 | 183 | 183 | AVAILABLE | N | N |
| `0336DCBAB05B9D5A` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `03AFDBD66E7929B1` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `03C6B06952C75089` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `060AD92489947D41` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `06138BC5AF602364` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `06409663226AF2F3` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `069059B7EF840F0C` | val | AML | Y | 163 | 106 | 106 | AVAILABLE | N | N |
| `06FEE55734DEFE2E` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `072B030BA126B2F4` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `0777D5C17D4066B8` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `07E1CD7DCA89A167` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `084B6FBB10729ED4` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `091D584FCED301B4` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `093F65E080A295F8` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `0A09C8844BA8F093` | train | AML | Y | 104 | 71 | 71 | AVAILABLE | N | N |
| `0AA1883C6411F787` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `0F28B5D49B3020AF` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `12ED127BACFF11D6` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `12F54ACB6F92C80E` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `1385974ED5904A43` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `140F6969D5213FD0` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `149E9677A5989FD3` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `14BFA6BB14875E45` | train | ALL | Y | 188 | 170 | 170 | AVAILABLE | N | N |
| `1679091C5A880FAF` | val | CML | Y | 317 | 284 | 284 | AVAILABLE | N | N |
| `17E62166FC8586DF` | train | CML | Y | 430 | 328 | 328 | AVAILABLE | N | N |
| `182BE0C5CDCD5072` | val | CML | Y | 316 | 291 | 291 | AVAILABLE | N | N |
| `19CA14E7EA6328A4` | train | CML | Y | 348 | 301 | 301 | AVAILABLE | N | N |
| `1A00B3B296CD48BA` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `1C383CD30B7C298A` | val | CML | Y | 173 | 145 | 145 | AVAILABLE | N | N |
| `1C9AC0159C94D8D0` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `1F0E3DAD99908345` | val | CML | Y | 273 | 174 | 174 | AVAILABLE | N | N |
| `202CB962AC59075B` | train | AML | Y | 167 | 117 | 117 | AVAILABLE | N | N |
| `26657D5FF9020D2A` | train | ALL | Y | 21 | 18 | 18 | AVAILABLE | N | N |
| `2723D092B63885E0` | test | Other | Y | 276 | 210 | 210 | AVAILABLE | N | N |
| `274AD4786C3ABCA6` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `2838023A778DFAEC` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `28DD2C7955CE9264` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `2A38A4A9316C49E5` | train | ALL | Y | 277 | 211 | 211 | AVAILABLE | N | N |
| `2A79EA27C279E471` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `2B24D495052A8CE6` | train | Other | Y | 6 | 2 | 2 | AVAILABLE | N | N |
| `2B44928AE11FB938` | val | Other | Y | 106 | 81 | 81 | AVAILABLE | N | N |
| `2EAB84C785399B2D` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `31FEFC0E570CB386` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `3295C76ACBF4CAAE` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `32BB90E8976AAB52` | test | ALL | Y | 25 | 9 | 9 | AVAILABLE | N | N |
| `335F5352088D7D9B` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `33E75FF09DD601BB` | test | CML | Y | 251 | 198 | 198 | AVAILABLE | N | N |
| `3416A75F4CEA9109` | train | CML | Y | 338 | 259 | 259 | AVAILABLE | N | N |
| `34173CB38F07F89D` | train | CML | Y | 230 | 163 | 163 | AVAILABLE | N | N |
| `3421557EEC62C83E` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `357AAD4942E92FCC` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `3636638817772E42` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `3644A684F98EA8FE` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `36660E59856B4DE5` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `37693CFC748049E4` | train | CML | Y | 254 | 203 | 203 | AVAILABLE | N | N |
| `37A749D808E46495` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `38B3EFF8BAF56627` | train | ALL | Y | 205 | 97 | 97 | AVAILABLE | N | N |
| `38DB3AED920CF82A` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `39059724F73A9969` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `398499203B70C029` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `3988C7F88EBCB58C` | test | AML | Y | 171 | 92 | 92 | AVAILABLE | N | N |
| `399C0FDBA7E789C6` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `3B80517688B93186` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `3B8A614226A953A8` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `3C59DC048E885024` | train | CML | Y | 261 | 227 | 227 | AVAILABLE | N | N |
| `3DEF184AD8F4755F` | train | AML | Y | 161 | 120 | 120 | AVAILABLE | N | N |
| `42A0E188F5033BC6` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `43EC517D68B6EDD3` | train | ALL | Y | 300 | 263 | 263 | AVAILABLE | N | N |
| `4443AEE183B279F7` | train | CML | Y | 54 | 39 | 39 | AVAILABLE | N | N |
| `44F683A84163B352` | train | ALL | Y | 201 | 159 | 159 | AVAILABLE | N | N |
| `45C48CCE2E2D7FBD` | val | CML | Y | 356 | 251 | 251 | AVAILABLE | N | N |
| `47397D3C4DB6F903` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `47D1E990583C9C67` | val | AML | Y | 34 | 26 | 26 | AVAILABLE | N | N |
| `4C56FF4CE4AAF957` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `4E732CED3463D06D` | train | CML | Y | 429 | 342 | 342 | AVAILABLE | N | N |
| `4F584FE84E3D2EDE` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `54229ABFCFA5649E` | val | ALL | Y | 120 | 86 | 86 | AVAILABLE | N | N |
| `5878A7AB84FB4340` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `58A2FC6ED39FD083` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `5C820C4E9439DC59` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `5EF059938BA799AA` | train | ALL | Y | 335 | 204 | 204 | AVAILABLE | N | N |
| `5F93F983524DEF3D` | train | Other | Y | 48 | 29 | 29 | AVAILABLE | N | N |
| `5FD0B37CD7DBBB00` | train | ALL | Y | 332 | 231 | 231 | AVAILABLE | N | N |
| `61BEED9E2C7DBF9C` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `621BF66DDB7C962A` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `623AE0B81861352D` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `6364D3F0F495B6AB` | train | CML | Y | 230 | 164 | 164 | AVAILABLE | N | N |
| `63DC7ED1010D3C3B` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `642E92EFB7942173` | val | CML | Y | 362 | 268 | 268 | AVAILABLE | N | N |
| `6512BD43D9CAA6E0` | val | CML | Y | 225 | 167 | 167 | AVAILABLE | N | N |
| `65B9EEA6E1CC6BB9` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `65DED5353C5EE48D` | train | AML | Y | 80 | 35 | 35 | AVAILABLE | N | N |
| `66F041E16A60928B` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `67C6A1E7CE56D3D6` | train | CML | Y | 630 | 531 | 531 | AVAILABLE | N | N |
| `68D30A9594728BC3` | val | ALL | Y | 388 | 249 | 249 | AVAILABLE | N | N |
| `6974CE5AC660610B` | train | ALL | Y | 252 | 129 | 129 | AVAILABLE | N | N |
| `698D51A19D8A121C` | train | ALL | Y | 35 | 29 | 29 | AVAILABLE | N | N |
| `69ADC1E107F7F7D0` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `6C4B761A28B734FE` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `6C8349CC7260AE62` | train | CML | Y | 370 | 280 | 280 | AVAILABLE | N | N |
| `6F3EF77AC0E3619E` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `6F4922F45568161A` | train | CML | Y | 245 | 139 | 139 | AVAILABLE | N | N |
| `700049AE9F8CB763` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `70EFDF2EC9B08607` | train | CML | Y | 487 | 417 | 417 | AVAILABLE | N | N |
| `72B32A1F754BA1C0` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `73278A4A86960EEB` | val | Other | Y | 246 | 171 | 171 | AVAILABLE | N | N |
| `74DB120F0A8E5646` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `75612D01831B0B8C` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `7647966B7343C290` | train | ALL | Y | 302 | 137 | 137 | AVAILABLE | N | N |
| `76DC611D6EBAAFC6` | val | AML | Y | 153 | 111 | 111 | AVAILABLE | N | N |
| `7A614FD06C325499` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `7CBBC409EC990F19` | test | ALL | Y | 259 | 176 | 176 | AVAILABLE | N | N |
| `7E7757B1E12ABCB7` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `7EF605FC8DBA5425` | train | Other | Y | 15 | 6 | 6 | AVAILABLE | N | N |
| `7F100B7B36092FB9` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `7F1DE29E6DA19D22` | train | AML | Y | 102 | 89 | 89 | AVAILABLE | N | N |
| `7F39F8317FBDB198` | train | ALL | Y | 544 | 394 | 394 | AVAILABLE | N | N |
| `7F6FFAA6BB0B4080` | train | Other | Y | 146 | 96 | 96 | AVAILABLE | N | N |
| `82161242827B703E` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `82AA4B0AF34C2313` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `84D9EE44E457DDEF` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `8613985EC49EB8F7` | train | ALL | Y | 454 | 333 | 333 | AVAILABLE | N | N |
| `87D5BE18C3F5EA97` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `8963A6178314866F` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `8D5E957F29789348` | train | AML | Y | 51 | 33 | 33 | AVAILABLE | N | N |
| `8E296A067A375633` | train | CML | Y | 376 | 297 | 297 | AVAILABLE | N | N |
| `8EB6D8747175919F` | unlabeled_extension | CML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `8F121CE07D74717E` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `8F14E45FCEEA167A` | train | CML | Y | 134 | 76 | 76 | AVAILABLE | N | N |
| `8F85517967795EEE` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `92CC227532D17E56` | train | ALL | Y | 234 | 91 | 91 | AVAILABLE | N | N |
| `93DB85ED909C1383` | test | ALL | Y | 62 | 24 | 24 | AVAILABLE | N | N |
| `93ED4838BE11F21D` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `9461E2E0784D432D` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `9766527F2B5D3E95` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `979D472A84804B9F` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `98DCE83DA57B0395` | train | ALL | Y | 271 | 241 | 241 | AVAILABLE | N | N |
| `98F13708210194C4` | train | CML | Y | 301 | 241 | 241 | AVAILABLE | N | N |
| `9A1158154DFA42CA` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `9A15F38C5291BA67` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `9B04D152845EC0A3` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `9B8619251A19057C` | train | AML | Y | 242 | 186 | 186 | AVAILABLE | N | N |
| `9BF31C7FF062936A` | train | CML | Y | 351 | 300 | 300 | AVAILABLE | N | N |
| `9DCB88E013764959` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `9F61408E3AFB633E` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `9FC3D7152BA9336A` | train | AML | Y | 191 | 122 | 122 | AVAILABLE | N | N |
| `A0A080F42E6F13B3` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `A1D0C6E83F027327` | train | CML | Y | 782 | 593 | 593 | AVAILABLE | N | N |
| `A3C65C2974270FD0` | test | ALL | Y | 508 | 335 | 335 | AVAILABLE | N | N |
| `A3F390D88E4C41F2` | train | ALL | Y | 71 | 47 | 47 | AVAILABLE | N | N |
| `A4A042CF4FD6BFB4` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `A5771BCE93E200C3` | train | CML | Y | 432 | 270 | 270 | AVAILABLE | N | N |
| `A597E50502F5FF68` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `A5BFC9E07964F8DD` | test | CML | Y | 361 | 283 | 283 | AVAILABLE | N | N |
| `A61D1A964CA038E6` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `A684ECEEE76FC522` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `A6BBC91AE73DD21C` | train | CML | Y | 413 | 241 | 241 | AVAILABLE | N | N |
| `A87FF679A2F3E71D` | train | CML | Y | 407 | 304 | 304 | AVAILABLE | N | N |
| `A8F15EDA80C50ADB` | train | Other | Y | 56 | 26 | 26 | AVAILABLE | N | N |
| `A97DA629B098B75C` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `AAB3238922BCC25A` | train | CML | Y | 381 | 291 | 291 | AVAILABLE | N | N |
| `AC627AB1CCBDB62E` | val | ALL | Y | 154 | 125 | 125 | AVAILABLE | N | N |
| `ACC68D832D90A3AA` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `AD61AB143223EFBC` | val | ALL | Y | 65 | 48 | 48 | AVAILABLE | N | N |
| `B1A59B315FC9A300` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `B1D10E7BAFA44212` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `B3E3E393C77E35A4` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `B4A67623E7A47A44` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `B53B3A3D6AB90CE0` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `B6D767D2F8ED5D21` | train | CML | Y | 378 | 262 | 262 | AVAILABLE | N | N |
| `B73CE398C39F506A` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `BA6484A665B9A691` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `BB8CF22800F0D97A` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `BCDCEF5EF4CF9871` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `BD4C9AB730F55132` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `BD686FD640BE98EF` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `BF8229696F7A3BB4` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `C0C7C76D30BD3DCA` | train | CML | Y | 364 | 224 | 224 | AVAILABLE | N | N |
| `C0E190D8267E3670` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `C16A5320FA475530` | test | CML | Y | 308 | 275 | 275 | AVAILABLE | N | N |
| `C20AD4D76FE97759` | test | CML | Y | 266 | 237 | 237 | AVAILABLE | N | N |
| `C24CD76E1CE41366` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `C24D731AC18944F0` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `C4CA4238A0B92382` | train | CML | Y | 136 | 109 | 109 | AVAILABLE | N | N |
| `C51CE410C124A10E` | train | CML | Y | 235 | 174 | 174 | AVAILABLE | N | N |
| `C52F1BD66CC19D05` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `C74D97B01EAE257E` | train | CML | Y | 173 | 139 | 139 | AVAILABLE | N | N |
| `C7E1249FFC03EB9D` | train | ALL | Y | 129 | 106 | 106 | AVAILABLE | N | N |
| `C7FF6173CE74C106` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `C81E728D9D4C2F63` | test | CML | Y | 297 | 235 | 235 | AVAILABLE | N | N |
| `C82561EC215A6E31` | test | CML | Y | 270 | 213 | 213 | AVAILABLE | N | N |
| `C8FFE9A587B126F1` | train | AML | Y | 120 | 94 | 94 | AVAILABLE | N | N |
| `C912541E67CD2244` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `C9F0F895FB98AB91` | train | CML | Y | 280 | 236 | 236 | AVAILABLE | N | N |
| `CA46C1B9512A7A83` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `CB70AB375662576B` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `D09BF41544A3365A` | train | ALL | Y | 138 | 75 | 75 | AVAILABLE | N | N |
| `D1C38A09ACC34845` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `D1F491A404D68548` | train | AML | Y | 42 | 28 | 28 | AVAILABLE | N | N |
| `D1FE173D08E95939` | train | ALL | Y | 103 | 85 | 85 | AVAILABLE | N | N |
| `D25E9132CF24B8A7` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `D2DDEA18F00665CE` | train | ALL | Y | 331 | 258 | 258 | AVAILABLE | N | N |
| `D3D9446802A44259` | train | CML | Y | 370 | 288 | 288 | AVAILABLE | N | N |
| `D645920E395FEDAD` | test | CML | Y | 228 | 161 | 161 | AVAILABLE | N | N |
| `D67D8AB4F4C10BF2` | train | CML | Y | 285 | 216 | 216 | AVAILABLE | N | N |
| `D82C8D1619AD8176` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `D96409BF89421768` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `D9D4F495E875A2E0` | train | CML | Y | 231 | 178 | 178 | AVAILABLE | N | N |
| `DA4FB5C6E93E74D3` | train | Other | Y | 396 | 258 | 258 | AVAILABLE | N | N |
| `E00DA03B685A0DD1` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `E2C0BE24560D78C5` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `E2C420D928D4BF8C` | test | ALL | Y | 340 | 278 | 278 | AVAILABLE | N | N |
| `E2EF524FBF3D9FE6` | train | ALL | Y | 291 | 244 | 244 | AVAILABLE | N | N |
| `E369853DF766FA44` | test | CML | Y | 390 | 297 | 297 | AVAILABLE | N | N |
| `E4A6222CDB5B3437` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `E4DA3B7FBBCE2345` | train | CML | Y | 354 | 273 | 273 | AVAILABLE | N | N |
| `E56954B4F6347E89` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `E96ED478DAB8595A` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `EA5D2F1C4608232E` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `EAA51AE97DBF78D2` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `EAE27D77CA20DB30` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `EB160DE1DE89D905` | val | ALL | Y | 456 | 368 | 368 | AVAILABLE | N | N |
| `EC5DECCA5ED3D6B8` | train | AML | Y | 203 | 83 | 83 | AVAILABLE | N | N |
| `EC8956637A99787B` | train | ALL | Y | 288 | 180 | 180 | AVAILABLE | N | N |
| `EC8CE6ABB3E952A8` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `ECCBC87E4B5CE2FE` | train | CML | Y | 283 | 231 | 231 | AVAILABLE | N | N |
| `ED3D2C21991E3BEF` | train | ALL | Y | 274 | 197 | 197 | AVAILABLE | N | N |
| `EDA80A3D5B344BC4` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `F033AB37C30201F7` | test | ALL | Y | 308 | 259 | 259 | AVAILABLE | N | N |
| `F0549529AF3E0E57` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `F0935E4CD5920AA6` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `F2217062E9A397A1` | test | AML | Y | 79 | 56 | 56 | AVAILABLE | N | N |
| `F4B9EC30AD9F68F8` | train | ALL | Y | 403 | 267 | 267 | AVAILABLE | N | N |
| `F7177163C833DFF4` | train | CML | Y | 305 | 238 | 238 | AVAILABLE | N | N |
| `F7405A633A5F7482` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `F768F6C7948E6A75` | unlabeled_extension | Other | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `F899139DF5E10593` | train | ALL | Y | 101 | 73 | 73 | AVAILABLE | N | N |
| `FBD7939D674997CD` | train | ALL | Y | 130 | 68 | 68 | AVAILABLE | N | N |
| `FC221309746013AC` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `FC490CA45C00B124` | unlabeled_extension | ALL | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |
| `FFD52F3C7E12435A` | unlabeled_extension | AML | N | 0 | 0 | 0 | ZERO_ANNOTATIONS | N | N |

---

## 5. Phase 2 — Crop Traceability

- SUM(crops per patient) = **20,852** == 20,852: PASS
- All 20,852 valid crops belong to the **113 consensus-annotated patients only**
- 132 unlabeled extension patients have **0 crops by design** (no cell-level annotations exist)
- 0 orphan crop directories
- 0 duplicate crop paths

---

*Audit complete.*
