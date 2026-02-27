export interface Category {
  code: 1 | 2 | 3 | 4;
  name: string;
  subtitle: string;
  note: string;
  accent: string;
}

export interface CatalogItem {
  id: string;
  page: 2 | 3 | 4 | 5;
  categoryCode: Category["code"];
  title: string;
  badge: string;
  modelCode: string;
  colour: number;
  colourName: string;
  location: 1 | 2 | 3 | 4 | 5 | 6;
  modelPhotography: 1 | 2;
  price: number;
}

export const PAGE_NUMBERS = [1, 2, 3, 4, 5] as const;

export const locationLabels: Record<CatalogItem["location"], string> = {
  1: "Top Left",
  2: "Top Middle",
  3: "Top Right",
  4: "Bottom Left",
  5: "Bottom Middle",
  6: "Bottom Right",
};

export const photographyLabels: Record<CatalogItem["modelPhotography"], string> = {
  1: "En face",
  2: "Profile",
};

export const categories: Category[] = [
  {
    code: 1,
    name: "Trousers",
    subtitle: "Comfort-first bottoms",
    note: "Stretch panels and tapered cuts for daily wear.",
    accent: "#ffd7c6",
  },
  {
    code: 2,
    name: "Skirts",
    subtitle: "Fluid movement",
    note: "Adaptive waists and office-friendly drape silhouettes.",
    accent: "#e1d7ff",
  },
  {
    code: 3,
    name: "Blouses",
    subtitle: "Layer-ready tops",
    note: "Nursing-compatible openings and breathable textures.",
    accent: "#d6f0de",
  },
  {
    code: 4,
    name: "Sale",
    subtitle: "Value highlights",
    note: "Promotions and bundles from prior seasonal drops.",
    accent: "#ffe6ad",
  },
];

const colourPalette: Array<{ code: number; name: string }> = [
  { code: 1, name: "beige" },
  { code: 2, name: "black" },
  { code: 3, name: "blue" },
  { code: 4, name: "brown" },
  { code: 5, name: "burgundy" },
  { code: 6, name: "gray" },
  { code: 7, name: "green" },
  { code: 8, name: "navy" },
  { code: 9, name: "multi" },
  { code: 10, name: "olive" },
  { code: 11, name: "pink" },
  { code: 12, name: "red" },
  { code: 13, name: "violet" },
  { code: 14, name: "white" },
];

const nameByCategory: Record<Category["code"], string[]> = {
  1: ["Cloudline", "LiftEase", "SoftForm", "UrbanFlex", "CalmCurve", "DailyDrape"],
  2: ["Bloomline", "MiraFlow", "Nesting", "SatinFold", "ArcMidi", "Boardwalk"],
  3: ["Luma", "NursingEase", "Halo", "Breeze", "Frame", "MetroSoft"],
  4: ["Capsule", "Outlet", "Bundle", "FinalTag", "SmartSave", "ValueEdit"],
};

const badgeByPage: Record<2 | 3 | 4 | 5, string> = {
  2: "Page 2 Featured",
  3: "Page 3 Trending",
  4: "Page 4 Essentials",
  5: "Page 5 Spotlight",
};

export const catalogItems: CatalogItem[] = categories.flatMap((category) =>
  ([2, 3, 4, 5] as const).flatMap((page) =>
    ([1, 2, 3, 4, 5, 6] as const).map((location, idx) => {
      const prefix = String.fromCharCode(64 + category.code);
      const modelCode = `${prefix}${page}${idx + 1}`;
      const palette = colourPalette[(category.code * page + idx) % colourPalette.length];
      const modelPhotography = (idx % 2 === 0 ? 1 : 2) as CatalogItem["modelPhotography"];

      return {
        id: `${category.code}-${page}-${location}`,
        page,
        categoryCode: category.code,
        title: `${nameByCategory[category.code][idx]} ${category.name}`,
        badge: badgeByPage[page],
        modelCode,
        colour: palette.code,
        colourName: palette.name,
        location,
        modelPhotography,
        price: 31 + category.code * 7 + page * 4 + idx * 2,
      };
    }),
  ),
);
