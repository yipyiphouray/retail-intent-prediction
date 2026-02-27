import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test } from "vitest";

import { App } from "../App";
import type { DemoApi } from "../api";

function buildApiMock(): DemoApi {
  let clickCount = 0;

  return {
    createSession: async () => ({
      session_id: 404,
      click_count: 0,
      status: "collecting",
      prediction: null,
    }),
    postClick: async () => {
      clickCount += 1;
      return {
        session_id: 404,
        click_count: clickCount,
        triggered: clickCount === 5,
        prediction:
          clickCount >= 5
            ? {
                label: "high-intent",
                probability: 0.9,
              }
            : null,
        show_ad: clickCount === 5,
        raw_row: {
          year: 2008,
          month: 4,
          day: 1,
          order: clickCount,
          country: 29,
          "session ID": 404,
          "page 1 (main category)": 1,
          "page 2 (clothing model)": "A21",
          colour: 3,
          location: 1,
          "model photography": 1,
          price: 46,
          "price 2": 2,
          page: 2,
        },
      };
    },
  };
}

test("supports category-to-product navigation and moves click analytics to dashboard", async () => {
  const apiMock = buildApiMock();
  render(<App api={apiMock} />);

  const categoryButton = await screen.findByRole("button", {
    name: /Browse Trousers/i,
  });

  await userEvent.click(categoryButton);

  const photoButton = await screen.findByRole("button", {
    name: /View model A21/i,
  });

  for (let i = 0; i < 5; i += 1) {
    await userEvent.click(photoButton);
  }

  expect(
    await screen.findByRole("heading", {
      name: /Complete the look with our Comfort\+ Nursing Pack/i,
    }),
  ).toBeInTheDocument();

  await userEvent.click(screen.getByRole("button", { name: /Analyst Dashboard/i }));

  await waitFor(() => {
    expect(screen.getByRole("heading", { name: /Session Behavior Dashboard/i })).toBeInTheDocument();
  });

  const clickMetric = screen.getByText("Clicks Captured").closest("article");
  expect(clickMetric).not.toBeNull();
  expect(within(clickMetric as HTMLElement).getByText("5")).toBeInTheDocument();
  expect(screen.getAllByRole("row")).toHaveLength(6);
});
