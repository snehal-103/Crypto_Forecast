from django.db import models

class Cryptocurrency(models.Model):
    name = models.CharField(max_length=100)  # Name of the cryptocurrency
    symbol = models.CharField(max_length=10)  # Symbol (e.g., BTC, ETH)
    price = models.FloatField()  # Current price
    date_fetched = models.DateTimeField(auto_now_add=True)  # Timestamp

    def __str__(self):
        return f"{self.name} ({self.symbol}) - {self.price}"
