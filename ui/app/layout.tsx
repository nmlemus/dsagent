import type { Metadata } from "next";
import Link from "next/link";

import "./globals.css";

export const metadata: Metadata = {
  title: "DSAgent",
  description: "Data science workflows, run by a team of personas",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="app">
          <header className="topbar">
            <Link href="/" className="wordmark">
              DS<span>agent</span>
            </Link>
            <span className="topbar-spacer" />
            <span className="topbar-note">Aiuda Labs</span>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}
