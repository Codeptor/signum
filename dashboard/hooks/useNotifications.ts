"use client";

import { useEffect, useCallback } from "react";

export function useNotifications() {
  // Request permission on mount
  useEffect(() => {
    if (
      typeof window !== "undefined" &&
      "Notification" in window &&
      Notification.permission === "default"
    ) {
      Notification.requestPermission();
    }
  }, []);

  const notify = useCallback((message: string, title: string = "Signum") => {
    if (typeof window === "undefined") return;

    if ("Notification" in window && Notification.permission === "granted") {
      new Notification(title, { body: message });
    } else if (
      "Notification" in window &&
      Notification.permission !== "denied"
    ) {
      Notification.requestPermission().then((perm) => {
        if (perm === "granted") {
          new Notification(title, { body: message });
        }
      });
    }
  }, []);

  return { notify };
}
