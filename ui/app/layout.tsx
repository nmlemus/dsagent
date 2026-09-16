import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DSAgent",
  description: "Data science workflows, run by a team of personas",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
