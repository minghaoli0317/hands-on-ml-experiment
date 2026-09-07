def test_package_imports():
    import market_volatility

    assert market_volatility is not None
    assert market_volatility.FORWARD_HORIZON == 20
