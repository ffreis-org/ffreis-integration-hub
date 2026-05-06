"""Locust HTTP load profile for /invocations."""

from __future__ import annotations

from locust import HttpUser, between, task


class InferenceUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task
    def invocations(self) -> None:
        self.client.post(
            "/invocations",
            data=b"1,2,3\n4,5,6\n",
            headers={"Content-Type": "text/csv", "Accept": "application/json"},
            name="POST /invocations",
        )
