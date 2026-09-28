# commodity_prices — World Bank Pink Sheet real commodity prices

| | |
|---|---|
| Config key | `commodity_prices` |
| Module | `src/data/sources/commodity_prices/source.py`, parsing in `commodity_prices/prices.py` |
| Steps | FETCH, PREPARE · no `REQUIRES` |
| In panel | no; consumed by `snl_mining` PREPARE (price shock) |

## What it is

The World Bank Commodity Markets "Pink Sheet" (CMO Historical Data, annual), sheet "Annual Prices
(Real)", in constant 2010 US dollars. No separate deflation is applied.

## Raw data (FETCH)

One `.xlsx` (`prices_url` in `data.yaml`). The URL embeds a content hash and changes roughly
monthly, so a FETCH 404 means `prices_url` needs updating from
<https://www.worldbank.org/en/research/commodity-markets>. `prices_path` points PREPARE at an
already-downloaded copy (`raw/commodity_prices/auxiliary/CMO-Historical-Data-Annual.xlsx`).

## Prepared output (PREPARE)

`prepared/commodity_prices/misc/commodity_prices.parquet`, with columns `commodity` (canonical
key, `src/data/sources/commodities.py`), `year`, `price_real`, `ln_price_real`.

## Analysis caveats

- Most minor commodities have no World Bank series. `commodities.WORLD_BANK_COLUMNS` maps them to
  `None`, and they contribute nothing to a mine's price shock, following Berman et al. (2017),
  who restrict to minerals with usable world prices.
- Prices are world prices, set globally. That is the exogeneity argument for the shift-share
  instrument.
